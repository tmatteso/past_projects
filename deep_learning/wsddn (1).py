import numpy as np
import torch
import torchvision
import torch.utils.data as data
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torchvision.ops import roi_pool, roi_align


class WSDDN(nn.Module):
    n_classes = 20
    classes = np.asarray([
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ])

    def __init__(self, classes=None):
        super(WSDDN, self).__init__()

        if classes is not None:
            self.classes = classes
            self.n_classes = len(classes)
            #print(classes)

        # TODO (Q2.1): Define the WSDDN model
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True), # was True
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), # False?
            #nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.roi_pool = roi_pool # hopefully instantiating here is correct -- no!
        self.classifier = nn.Sequential(
            nn.Linear(9216, 4096), # what are the dims here? same as AlexNet?
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )
        # input to both fcs is class num x num of roi proposals
        self.score_fc = nn.Sequential(
            nn.Linear(4096, self.n_classes),
            nn.Softmax(dim=1), # take this one wrt class
         ) 
        self.bbox_fc = self.score_fc = nn.Sequential(
            nn.Linear(4096, self.n_classes),
            nn.Softmax(dim=0), # take this one wrt roi proposals
         ) 
        # you also need to combine the scores somewhere -- just take the hadamard prod of score and bbox output in forward -- (N_boxes x 20)
        # loss  for what part -- this really just seems like BCE with logits as before, why do we have three sep loss things?
        self.cross_entropy = None #torch.nn.BCEWithLogitsLoss() # this instation is highly sus, I would init with None

    @property
    def loss(self):
        return self.cross_entropy

    def forward(self,
                image,
                rois=None,
                gt_vec=None,
                ):
        x = self.features(image)
        # batch dim will always be one, reshape to Boxesx4
        #print(x.shape)
        #print(rois.shape)
        # box coordinates in (x1, y1, x2, y2) -- needed by roi pool
        # proposal come in [y_min, x_min, y_max, x_max] -- changed this to (x1, y1, x2, y2) nowa
        #rois = rois.reshape(rois.shape[1],rois.shape[2])
        rois = list(rois) # has to be a list with Lx4 tensors in it according to docs
        #print(len(rois), rois[0].shape)
        #output_size = (6,6) is said to be fine according to TAs 1/16?
        roi_pool = self.roi_pool(x, rois, output_size=6, spatial_scale=31) # needs input AND boxes, outputs
        
        flattened = roi_pool.view(len(roi_pool),-1) 
        #print(flattened.shape) 
        #raise Exception
        #print(flattened.shape)
        #raise Exception
        x = self.classifier(flattened)
        #print(x.shape)
        scores = self.score_fc(x)
        #print(scores.shape)
        bbox = self.bbox_fc(x)
        #print(bbox.shape)
        # take the hadamard prod of score and bbox output
        combined = torch.mul(scores, bbox) # generally there is a sum here too
        #print(combined.shape)
        # TODO (Q2.1): Use image and rois as input
        # compute cls_prob which are N_roi X 20 scores -- why? for loss calculation -- where do these come from? combined
        cls_prob = combined

        # during training the loss used is different?
        if self.training:
            #print(gt_vec) # list of tensors with class names (NOT one hot)
            label_vec = gt_vec.view(self.n_classes, -1)
            self.cross_entropy = self.build_loss(cls_prob, label_vec)
        return cls_prob

    def build_loss(self, cls_prob, label_vec):
        """Computes the loss

        :cls_prob: N_roix20 output scores
        :label_vec: 1x20 one hot label vector
        :returns: loss

        """
        # TODO (Q2.1): Compute the appropriate loss using the cls_prob -- this is the loss for the classification part 
        # that is the output of forward()
        # Checkout forward() to see how it is called -- seems like I compute the sum here
        # sum is in the R dim, check dims of combined -- R by C
        output = torch.sum(cls_prob, dim=0) # now it is 1x20 # maybe try with BCE if it doesn't 
        output = output.view(output.shape[0], 1) 
        output = torch.clamp(output, 0.0, 1.0)
        #print(label_vec.shape, output.shape)
        cross_entropy =  torch.nn.BCELoss(reduction='sum') # apparently without this it is trash
        loss = cross_entropy(output, label_vec) # this is the BCE with logits loss -- it needs (target, output)

        return loss
