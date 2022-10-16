import argparse
from operator import imul
import os
import shutil
import time

import matplotlib.pyplot as plt
import sklearn
import sklearn.metrics
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
#import torchvision.transforms.functional as TF
#import torchvision.transforms as T
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.utils as utils
import wandb

from AlexNet import localizer_alexnet, localizer_alexnet_robust
from voc_dataset import *
from utils import *


# GLOBAL MAX POOLING happens in train loop, not forward!

#  FIXING THE RANDOM SEEDS BEFORE CREATING DATALOADERS.

USE_WANDB = True  # use flags, wandb is not convenient for debugging
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', default='localizer_alexnet_robust')
parser.add_argument(
    '-j',
    '--workers',
    default=4,
    type=int,
    metavar='N',
    help='number of data loading workers (default: 4)')
parser.add_argument(
    '--epochs',
    default=30,# 2 for 1.3, 30 for 1.6
    type=int,
    metavar='N',
    help='number of total epochs to run')
parser.add_argument(
    '--start-epoch',
    default=0,
    type=int,
    metavar='N',
    help='manual epoch number (useful on restarts)')
parser.add_argument(
    '-b',
    '--batch-size',
    default=32, #try 1, should be 32 once collate works
    type=int,
    metavar='N',
    help='mini-batch size (default: 256)')
parser.add_argument(
    '--lr',
    '--learning-rate',
    default=0.1,
    type=float,
    metavar='LR',
    help='initial learning rate')
parser.add_argument(
    '--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument(
    '--weight-decay',
    '--wd',
    default=1e-4,
    type=float,
    metavar='W',
    help='weight decay (default: 1e-4)')
parser.add_argument(
    '--print-freq',
    '-p',
    default=10,
    type=int,
    metavar='N',
    help='print frequency (default: 10)')
parser.add_argument(
    '--eval-freq',
    default=2,
    type=int,
    metavar='N',
    help='print frequency (default: 10)')
parser.add_argument(
    '--resume',
    default='',
    type=str,
    metavar='PATH',
    help='path to latest checkpoint (default: none)')
parser.add_argument(
    '-e',
    '--evaluate',
    dest='evaluate',# when store false, my loss does not nan
    action='store_true', # store_false for rn, true for when you want validation
    help='evaluate model on validation set')
parser.add_argument(
    '--pretrained',
    dest='pretrained',
    action='store_true',# I changed it to store_true, was store_false before
    help='use pre-trained model')
parser.add_argument(
    '--world-size',
    default=1,
    type=int,
    help='number of distributed processes')
parser.add_argument(
    '--dist-url',
    default='tcp://224.66.41.62:23456',
    type=str,
    help='url used to set up distributed training')
parser.add_argument(
    '--dist-backend', default='gloo', type=str, help='distributed backend')
parser.add_argument('--vis', action='store_true')

best_prec1 = 0

# It seems like I just rewrite this file between 1.3 and 1.6! -- lol why not use more args to reuse
def main():
    global args, best_prec1
    args = parser.parse_args()
    args.distributed = args.world_size > 1

    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch == 'localizer_alexnet':
        model = localizer_alexnet(pretrained=True)#changed from args.pretrained
    elif args.arch == 'localizer_alexnet_robust':
        model = localizer_alexnet_robust(pretrained=args.pretrained)
    print(model)

    model.features = torch.nn.DataParallel(model.features)
    model.cuda()

    # TODO (Q1.1): define loss function (criterion) and optimizer from [1]
    # also use an LR scheduler to decay LR by 10 every 30 epochs -- this is the 1.6 version 
    
    # k is for each class -- be careful about if this returns sigmoid or not in the forward
    criterion = torch.nn.BCEWithLogitsLoss()#reduction='mean') # but this should be summed over for all classes
    
    # tried changing this, did not help
    # need the logits output-- this will produce the raw logits output before the sigmoid as desired
    # it doesn't matter, keep as recution = mean for now, this will absorb the mean into the learning rate, but it shouldn't make a difference in the end. 
    # batchsize=32
    
    # another possible loss would be soft margin loss
    # optimization from paper
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # optimizer for Q1.3 -- tried 0.05, it's the best so far
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9) # 0.01, tried 0.1, 0.01, 0.001 all explode
    # desired scheduler add cond with epoch size
    # scheduler = StepLR(optimizer, step_size=30, gamma=0.1) # scheduler is not until later

# use train dataset index instead of the train dataloader with shuffle = False to make the batches truely deterministic

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code

    # TODO (Q1.1): Create Datasets and Dataloaders using VOCDataset
    # Ensure that the sizes are 512x512 -- pretty sure collation is needed here
    # Also ensure that data directories are correct
    # The ones use for testing by TAs might be different
    train_dataset = VOCDataset('trainval') # trainval 
    val_dataset = VOCDataset('test') # test? -- Yes!
    torch.manual_seed(1)    
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    # we do use a built in dataloader, we definitely do need a collate fn
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        shuffle=False, #was False before 
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    # TODO (Q1.3): Create loggers for wandb.
    # Ideally, use flags since wandb makes it harder to debug code.
    if USE_WANDB:
        wandb.init(project="vlr-hw1", reinit=True)
    

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)
        # evaluate on validation set
        if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
            m1, m2 = validate(val_loader, model, criterion, epoch)

            score = m1 * m2
            # remember best prec@1 and save checkpoint
            is_best = score > best_prec1
            best_prec1 = max(score, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best)

# this has to be debugged -- highest priority, need to see if the heatmaps aren't shit
def get_images(data, imoutput):
    transform1 = transforms.ToPILImage() # assuming your image in x
    #.show()#.save("raw.jpg")
    # use make grid from torchvision
    dataset = VOCDataset('trainval')
    class_id_to_label = dict(enumerate(dataset.CLASS_NAMES)) # this method is too brittle, choose class random
    ld = [0,1]
    image_arr = []
    #heatmap_arr = []
    #print(imoutput.shape)
    for i in ld: # need to get the right heatmap for a clas in the image
        class_to_see = data["gt_classes"][i][0] # always grab the first class that appears
        #print(class_to_see)
        img = (data["image"][i]) #transform1
        #image_arr.append(img)
        sigmoid = nn.Sigmoid() 
        normal = sigmoid(imoutput[i][class_to_see]) 
        normal = imoutput[i][class_to_see]
        #print(normal.shape)
        #
        input = normal.reshape(1,1, normal.shape[0],normal.shape[1]) #1,1
        #print(input.shape)
        upscaled = F.interpolate(input, size=data["image"][i].shape[1:], mode="bilinear") # cv2  try bilinear interpolation
        #print(upscaled.shape)
        
        upscaled = upscaled.reshape(data["image"][i].shape[1:])
        #print(upscaled.shape)
        
        upscaled = upscaled.cpu().detach().numpy()
        name = class_id_to_label[class_to_see] 
        #print(name)
        #raise Exception
        wandb.log({"normal_image_" + name: wandb.Image(img)})
        # get's first image in batch's first class heatmap
        heat_image = plt.imshow(upscaled, cmap='jet', interpolation='bilinear')
        image_arr.append(heat_image)
        wandb.log({"heatmap_" + name: wandb.Image(heat_image)}) #transform1(upscaled))})
        #raise Exception
    #utils.make_grid(nrow =2)


# TODO: You can add input arguments if you wish
def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()
    # it will do this one epoch at a time
    # switch to train mode
    model.train()

    end = time.time()
    for i, (data) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        #print(data["image"])
        #imoutput = model(data["image"])
        #print(imoutput)
        #raise error
        # TODO (Q1.1): Get inputs from the data dict
        # Convert inputs to cuda if training on GPU
        images = data["image"].to(torch.device('cuda:0'))
        #print(len(data["image"])) # it is of the right batch size!
        #raise Exception
        target = data["label"].to(torch.device('cuda:0'))
        imoutput = model(images) # model is returning nans
        # input image is fine 
        # apply the global max pool here
        #print(images.shape)
        #print(imoutput) # now this is 29x29?? # you removed the max pool at the end of features
        #imoutput = F.max_pool2d(imoutput, kernel_size=imoutput.size()[2:]) -- no max pool for alexnet robust
        #print(imoutput.shape)
        imoutput = F.avg_pool2d(imoutput, kernel_size=imoutput.size()[2:])
        #print(imoutput.shape)
        #print(data["image"])
        #print(imoutput)
        #print(target.shape)
        #print(target)
        # TODO (Q1.1): Get output from model -- this is the heatmap
        # TODO (Q1.1): Perform any necessary operations on the output
        #sigmoid = nn.Sigmoid() -- still nans even after disabling this
        #imoutput = sigmoid(imoutput)
        #print(target.shape)
        #print(imoutput.shape)
        imoutput = torch.reshape(imoutput, (imoutput.shape[0], imoutput.shape[1]))
        #print(imoutput)
        #print(imoutput.shape)
        # TODO (Q1.1): Compute loss using ``criterion``
        loss = criterion(imoutput, target) # was in opposite order before
        #print(loss.item())
        #print(imoutput.size(0))
        #if i ==1:
        #   raise Exception
        #raise Exception
        # measure metrics and record loss
        m1 = metric1(imoutput.data, target)#
        m2 = metric2(imoutput.data, target)#
        losses.update(loss.item(), imoutput.size(0)) # The image you load using the dataloader is what input expects here
        #print(losses)
        #raise error
        avg_m1.update(m1)
        avg_m2.update(m2)
        #print(loss.item(), "hi")
        # TODO (Q1.1): compute gradient and perform optimizer step
        optimizer.zero_grad()
        loss.backward() # seems to be something in the update
        #raise error
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
                  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
                      epoch,
                      i,
                      len(train_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      avg_m1=avg_m1,
                      avg_m2=avg_m2))

        # TODO (Q1.3): Visualize/log things as mentioned in handout at appropriate intervals
        if USE_WANDB:
            wandb.log({'train/loss': loss})
            wandb.log({'train/m1': m1})
            wandb.log({'train/m2': m2})
        # Use wandb to plot images and the rescaled heatmaps (to image resolution) for any GT class for 2 images at epoch 0 and epoch 1.
        # images and corresponding GT label should be the same across epochs so you can monitor how the network is learning to localize objects.

        

        # End of train() -n it never seems to learn dogs very well -- it seems like it's just trying to use the darkest pixels!

# Recommended training loss at the end of training: ~(0.15, 0.20) (for 1.3 version)
def validate(val_loader, model, criterion, epoch=0):
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()

    # switch to evaluate mode -- just train longer??
    model.eval()

    end = time.time()
    for i, (data) in enumerate(val_loader):
        # the dataloader worries we the most
        # TODO (Q1.1): Get inputs from the data dict
        # Convert inputs to cuda if training on GPU
        target = data["label"].to(torch.device('cuda:0'))
        imoutput = model(data["image"])

        # TODO (Q1.1): Get output from model
        #imoutput = None  # how to get the heatmap?
        if i == 0 and USE_WANDB:
            get_images(data, imoutput)
        # TODO (Q1.1): Perform any necessary functions on the output
        print(imoutput.shape)
        imoutput = F.max_pool2d(imoutput, kernel_size=imoutput.size()[2:])
        print(imoutput.shape)
        #print(target.shape)
        #print(imoutput.shape)
        imoutput = torch.reshape(imoutput, (imoutput.shape[0], imoutput.shape[1]))
        # TODO (Q1.1): Compute loss using ``criterion``
        loss = criterion(target, imoutput)
        #print(loss.item())
        #raise error
        # measure metrics and record loss
        m1 = metric1(imoutput.data, target)
        m2 = metric2(imoutput.data, target)
        losses.update(loss.item(), imoutput.size(0))
        avg_m1.update(m1)
        avg_m2.update(m2)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
                  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
                      i,
                      len(val_loader),
                      batch_time=batch_time,
                      loss=losses,
                      avg_m1=avg_m1,
                      avg_m2=avg_m2))

        # TODO (Q1.3): Visualize things as mentioned in handout  -- ther is no val in 1.3?
        # TODO (Q1.3): Visualize at appropriate intervals
        # We also want to visualize where the network is "looking" for objects during 
        # classification. We can use the model's outputs for this. 
        #Use wandb to plot images and the rescaled heatmaps (to image resolution) for any GT 
        #class for 2 images at epoch 0 and epoch 1. The images and corresponding GT label 
        # should be the same across epochs so you can monitor how the network is learning to 
        #localize objects. (Hint: a heatmap has values between 0 and 1 while the model output 
        # does not!) -- ensure that they only want the heatmap in val
        if USE_WANDB:
            #wandb.log({'train/loss': loss})
            wandb.log({'val/m1': m1})
            wandb.log({'val/m2': m2})

    print(' * Metric1 {avg_m1.avg:.3f} Metric2 {avg_m2.avg:.3f}'.format(
        avg_m1=avg_m1, avg_m2=avg_m2))

    return avg_m1.avg, avg_m2.avg
# the collate can be pretty trivial, do have to pad when you have a mismatch in the num of boxes, plus you still need to serve a batch of images

# TODO: You can make changes to this function if you wish (not necessary)
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
# write the metrics for rn 
        
# output size from the model needs to be pooled as explained in the further questions to get a (1,20,1,1) tensor        
# You can use PIL to create the heatmap and wandb for plotting.
# You could apply sigmoid to the model output to construct the heatmap.

# You can set shuffle = False and then plot certain indices from the training set. While this is not ideal, this is the only way we can ensure we are tracking the same image across epochs.
# this for the train loader and ensure that we are seeing the same images in order


# You can improve performance further by looking into the "weight" data item in the VOCDataset class. For each element in the batch, this tensor indicates whether the labels are valid or not. If there are no valid labels for a given class, you can skip that class
def metric1(output, target):
    # TODO (Q1.5): compute metric1 -- mAP
    #  While computing metric1, do not include classes which don't have a corresponding label present in the batch while computing the mean.
    # apply a Sigmoid on the maxpooled output and compute AP in a vectorized way. Ignore the classes that have no ground truth samples in a given batch.
    #  For Average Precision, sklearn computes this internally at various thresholds under the hood so you don't need to do it yourself. But for recall, you should binarize it
    # 1. Try converting both the GT and model outputs to float32.
    # 2. Note that the GT is the first argument and the output is the second argument.
    # 3. If you are using average_precision_score, you'd want to set average to None (assuming you are looping through each class).
    # So when computing metric1, for a class/label that is not present, what happens when
    # 1. pred = 0 :  skip it and not use it for computing the mean
    # pred = 1 : it should be 0 precision here
    # loop over classes, and calling average_precision_score on the corresponding target and output with average = None.
# I ignore the class when it does not have ground truth samples (I check if there is a 1)
# I’ve also tried to use Sigmoid on the output here as pointed out in the thread above.
# then I take the mean of all the scores and return that
    sigmoid = nn.Sigmoid() #32x20
    normal = sigmoid(output).cpu().detach().numpy()
    #print(normal)
    target = target.cpu().detach().numpy() # this obvs wrong
    #print(target)
    good_indices = np.sum(target, axis = 0) # sum across all samples for each class, if the clas does not appear it is 0
    good_indices = (good_indices != 0) *1 
    #print(good_indices.sum()) # 20 long binary array
    #raise Exception
    only_these = ~np.all(np.concatenate([target, normal]), axis=0)
    precisions = sklearn.metrics.average_precision_score(target[:,only_these], normal[:,only_these], average='samples')
    # precisions = []
    # # change the range to be only the number of labels that appear
    # for j in range(normal.shape[1]):
    #     if good_indices[j]:
    #         #print(target[:, j])
    #         #raise Exception
    #         precisions.append(sklearn.metrics.average_precision_score(target[:, j], normal[:, j], average = None))

    # # no gt -- filter o
    # # if no gt but pred is firing for these classes
    # # if pred appears in class and gtb does not, add it back
    # # if neither gt nor pred occur in the minibatch, do not add it back!
    # #print((precisions))
    # #raise Exception
    # precisions = np.mean(precisions) #+ 0.2 #(too cheeky?)
    #print(precisions)
    #for i in range(results.shape[0]):
        #for j in range(results.shape[1]):
            #if (results[i][j])
            #raise Exception
        #precisions.append(sklearn.metrics.average_precision_score(results[i], target[i], average = None))# 32, 20 for both
        #raise Exception
    return(precisions)
    #raise Exception
    #return 0


def metric2(output, target):
    # TODO (Q1.5): compute metric2 -- Recall
    # it would appear we need to apply a sigmoid to the model output to calc here
    # do include la
    # if GT has no label, but model predicts a label there, precision is 0. Thus, one must account for that in the final value of average precision accordingly 
    # you can’t just “remove” all classes without a GT label
    #  threshold the output while passing it to sklearn.metrics.recall_score and 0.5 is a reasonable value to use.
    # 1. You don't need to loop through all classes, recall_score can handle multi-label classification.
    # 2. Use average="samples".

    # I take the sigmoid of output, put a threshold of 0.5 on it, consider only those classes which have at 
    # least 1 sample with the ground truth of that class, and pass such filtered vectors of target and 
    # output to sklearn metrics for recall_score. 
    sigmoid = nn.Sigmoid()
    normal = sigmoid(output).cpu().detach().numpy()
    #print(normal[:5])
    #raise Exception
    threshold = np.array([0.4]) # fiddle with the threshold
    thresholded = (normal>threshold)#*1
    #print(normal)
    target = target.cpu().detach().numpy()
    new_normal, new_target = [], []
    good_indices = np.sum(target, axis = 0)
    good_indices = np.array((good_indices != 0) * 1) #wgt shenanigans??
    for j in range(normal.shape[1]):
        if good_indices[j]:
            new_normal.append(thresholded[:,j])
            new_target.append(target[:, j])

    new_normal = np.array(new_normal).T
    new_target = np.array(new_target).T
    #print(new_normal)
    #print(new_target)
     #.to(torch.device('cuda:0'))
    #results = (new_normal>threshold)*1
    #print(new_normal)
    recall = sklearn.metrics.recall_score(target, thresholded, average="micro") # this is after talking to the Prof -- TA said yes but samples
    #print(recall)
    #recall  = (sklearn.metrics.recall_score(target, thresholded, average="samples")) # this is what was said on piazza
    #raise Exception
    return recall


if __name__ == '__main__':
    main()
