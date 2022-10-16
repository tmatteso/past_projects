import torch.nn as nn
import torchvision.models as models
import torch
import torchvision.transforms as T

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

# need to use paper's loss function and optimizer. -- not alexnet but the other one
def weights_init(m):
    # in place change or do I need to retun the model?
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight.data)
        
# localizrr AlexNet will not be that similar to alexnet, all the linear layers will be conv2d
# must re init  the model, can't just take from the download. Do need the weigts though
        
# read the paper to make sure the ordering of your sigmoid and final max pool are in the right place. 

# is pasting alexnet source code really the best option??
class LocalizerAlexNet(nn.Module):
    def __init__(self, num_classes=20):
        super(LocalizerAlexNet, self).__init__()
        #dropout =
        # I don't think there was ever a batch norm
        # the ceil mode and dilation are by default
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
        # you really need to talk to TAs now, just have it ready for after SD
        #self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            #nn.Dropout(p=dropout),
            #nn.Linear(256 * 6 * 6, 4096),
            nn.Conv2d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=dropout),
            #nn.Linear(4096, 4096),
            nn.Conv2d(256, 256, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            #nn.Linear(4096, num_classes),
            nn.Conv2d(256, 20, kernel_size=1, stride=1),
        )
        
        # train the model using batchsize=32, learning rate=0.01, epochs=2
    def forward(self, x):
        # TODO (Q1.1): Define forward pass
        # weights explode after gradient update
        
        x = self.features(x)
        #print(x) # the first loss calc is fine, it must be the criterion or the optim
        #x = self.avgpool(x)
        #x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class LocalizerAlexNetRobust(nn.Module):
    def __init__(self, num_classes=20): # same thing but apply gaussian blur at end?
        super(LocalizerAlexNetRobust, self).__init__()
        # TODO (Q1.7): Define model
        #dropout =
        # I don't think there was ever a batch norm
        # the ceil mode and dilation are by default
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
        # you really need to talk to TAs now, just have it ready for after SD
        #self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            #nn.Linear(256 * 6 * 6, 4096),
            nn.Conv2d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            #nn.Linear(4096, 4096),
            nn.Conv2d(256, 256, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            #nn.Linear(4096, num_classes),
            nn.Conv2d(256, 20, kernel_size=1, stride=1),
        )
        self.blur = T.GaussianBlur(kernel_size= 5)

    def forward(self, x):
        # TODO (Q1.7): Define forward pass
        x = self.features(x)
        #print(x) # the first loss calc is fine, it must be the criterion or the optim
        #x = self.avgpool(x)
        #x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.blur(x)
        return x


def localizer_alexnet(pretrained=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # model is defined, how do I feed weights into the model? --ask Young Ge
    # Initialize the model from ImageNet (till the conv5 layer)
    # Initialize the rest of layers with Xavier initialization
    print("called")
    new_model = LocalizerAlexNet(**kwargs)
    mask_model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
    #weights = models.AlexNet_Weights.IMAGENET1K_V1
    #mask_model = models.alexnet(weights = weights)
    #print(mask_model.state_dict())
    # there's something wrong here
    print(pretrained)
    if pretrained:
        #print(pretrained)
        # how do I freeze the 
        with torch.no_grad():
            for layer in new_model.state_dict():
                #print(layer)
                if "features" in layer:
                    print(layer, "features")
                    new_model.state_dict()[layer].data = torch.nn.Parameter(
                        mask_model.state_dict()[layer].data)
                    #new_model.state_dict()[layer].requires_grad = False -- try this next
                elif "weight" in layer:
                    #pass
                    #print(layer, "clas weights") # apparently they want this instead ?? xavier_normal_
                    torch.nn.init.xavier_normal_(new_model.state_dict()[layer].data) # uniform normal 
                elif "bias" in layer:
                    #pass #trying the default init -- xavier normal is definitely better here
                    #print(layer, "clas bias")
                    new_model.state_dict()[layer].data.fill_(0.0)
        #print(new_model.state_dict())
                    #print(new_model.state_dict()[layer].data)
        #print(new_model.features[0].weight) # 0, 3, 6, 8, 10
        #print(new_model.features[3].weight) 
        #print(new_model.features[6].weight) 
        #print(new_model.features[8].weight) 
        #print(new_model.features[10].weight) 
        
        #torch.nn.init.xavier_uniform_(new_model.classifier[0].weight)
        #torch.nn.init.xavier_uniform_(new_model.classifier[2].weight)
        #torch.nn.init.xavier_uniform_(new_model.classifier[4].weight)
        #print(new_model.classifier[0].weight)
        #print(new_model.classifier[2].weight)
        #print(new_model.classifier[4].weight)
        # LocalizerAlexNet(**kwargs, weights)
        return new_model
    else:
        torch.nn.init.xavier_uniform_(new_model.features[0].weight)
        torch.nn.init.xavier_uniform_(new_model.features[3].weight)
        torch.nn.init.xavier_uniform_(new_model.features[6].weight)
        torch.nn.init.xavier_uniform_(new_model.features[8].weight)
        torch.nn.init.xavier_uniform_(new_model.features[10].weight)
        
        torch.nn.init.xavier_uniform_(new_model.classifier[0].weight)
        torch.nn.init.xavier_uniform_(new_model.classifier[2].weight)
        torch.nn.init.xavier_uniform_(new_model.classifier[4].weight)
    
    # TODO (Q1.3): Initialize weights based on whether it is pretrained or not

    return new_model


def localizer_alexnet_robust(pretrained=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    new_model = LocalizerAlexNet(**kwargs)
    mask_model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
    # TODO (Q1.7): Initialize weights based on whether it is pretrained or not
    if pretrained:
        # how do I freeze the 
        with torch.no_grad():
            for layer in new_model.state_dict():
                print(layer)
                if "features" in layer:
                    #print(layer)
                    new_model.state_dict()[layer].data = torch.nn.Parameter(
                        mask_model.state_dict()[layer].data)
                    new_model.state_dict()[layer].requires_grad = False 
                elif "weight" in layer:
                    torch.nn.init.xavier_normal_(new_model.state_dict()[layer].data) # was x uniform before
                elif "bias" in layer:
                    new_model.state_dict()[layer].data.fill_(0.0)
        #print(new_model.state_dict())
                    #print(new_model.state_dict()[layer].data)
        #print(new_model.features[0].weight) # 0, 3, 6, 8, 10
        #print(new_model.features[3].weight) 
        #print(new_model.features[6].weight) 
        #print(new_model.features[8].weight) 
        #print(new_model.features[10].weight) 
        
        #torch.nn.init.xavier_uniform_(new_model.classifier[0].weight)
        #torch.nn.init.xavier_uniform_(new_model.classifier[2].weight)
        #torch.nn.init.xavier_uniform_(new_model.classifier[4].weight)
        #print(new_model.classifier[0].weight)
        #print(new_model.classifier[2].weight)
        #print(new_model.classifier[4].weight)
        # LocalizerAlexNet(**kwargs, weights)
        return new_model
    else:
        torch.nn.init.xavier_uniform_(new_model.features[0].weight)
        torch.nn.init.xavier_uniform_(new_model.features[3].weight)
        torch.nn.init.xavier_uniform_(new_model.features[6].weight)
        torch.nn.init.xavier_uniform_(new_model.features[8].weight)
        torch.nn.init.xavier_uniform_(new_model.features[10].weight)
        
        torch.nn.init.xavier_uniform_(new_model.classifier[0].weight)
        torch.nn.init.xavier_uniform_(new_model.classifier[2].weight)
        torch.nn.init.xavier_uniform_(new_model.classifier[4].weight)
        return new_model
