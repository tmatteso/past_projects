''' --------------------------------------------------------
Written by Yufei Ye (https://github.com/JudyYe)
Edited by Anirudh Chakravarthy (https://github.com/anirudh-chakravarthy)
-------------------------------------------------------- '''
import os
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import scipy.io
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class VOCDataset(Dataset):
    CLASS_NAMES = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
        'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    INV_CLASS = {}

    for i in range(len(CLASS_NAMES)):
        INV_CLASS[CLASS_NAMES[i]] = i

    # TODO: Ensure data directory is correct
    def __init__(
        self,
        split='trainval',
        image_size=512,# 224 was original, before I had it at 512
        top_n=300, # 300 was original
        data_dir='data/VOCdevkit/VOC2007/'
    ):
        super().__init__()
        self.split = split     # 'trainval' or 'test'
        self.data_dir = data_dir
        self.size = image_size
        self.top_n = top_n      # top_n: number of proposals to return

        self.img_dir = os.path.join(data_dir, 'JPEGImages')
        self.ann_dir = os.path.join(data_dir, 'Annotations')
        self.selective_search_dir = os.path.join(
            data_dir, 'selective_search_data')
        self.roi_data = scipy.io.loadmat(
            self.selective_search_dir + '/voc_2007_' + split + '.mat')

        split_file = os.path.join(data_dir, 'ImageSets/Main', split + '.txt')
        with open(split_file) as fp:
            self.index_list = [line.strip() for line in fp]

        self.anno_list = self.preload_anno()

    @classmethod
    def get_class_name(cls, index):
        """
        :return: category name for the corresponding class index.
        """
        return cls.CLASS_NAMES[index]

    @classmethod
    def get_class_index(cls, name):
        """
        :return: class index for the corresponding category name.
        """
        return cls.INV_CLASS[name]

    def __len__(self):
        return len(self.index_list)

    def preload_anno(self):
        """
        :return: a list of labels.
        each element is in the form of [class, weight, gt_class_list, gt_boxes]
         where both class and weight are arrays/tensors in shape of [20],
         gt_class_list is a list of the class ids (separate for each instance)
         gt_boxes is a list of [xmin, ymin, xmax, ymax] values in the range 0 to 1
        """

        # TODO: Make sure you understand how the GT boxes and class labels are loaded
        label_list = []

        for index in self.index_list:
            fpath = os.path.join(self.ann_dir, index + '.xml')
            tree = ET.parse(fpath)
            root = tree.getroot()

            C = np.zeros(20)
            W = np.ones(20) * 2 # default to enable 1 or 0 later for difficulty

            # image h & w to normalize bbox coords
            height = 0
            width = 0

            # new list for each index
            gt_class_list = []
            gt_boxes = []

            for child in root:

                if child.tag == 'size':
                    width = int(child[0].text)
                    height = int(child[1].text)

                if child.tag == 'object':
                    C[self.INV_CLASS[child[0].text]] = 1    # item at index of child name become 1
                    if child[3].text == '1' and W[self.INV_CLASS[child[0].text]] == 2:
                        W[self.INV_CLASS[child[0].text]] = 0    # if not difficult, weight is one
                    elif child[3].text == '0' :
                        W[self.INV_CLASS[child[0].text]] = 1

                    # add class_index to gt_class_list
                    gt_class_list.append(self.INV_CLASS[child[0].text])

                    for t in child:
                        if t.tag == 'bndbox':
                            xmin = int(t[0].text) / width
                            ymin = int(t[1].text) / height
                            xmax = int(t[2].text) / width
                            ymax = int(t[3].text) / height
                            gt_boxes.append([xmin, ymin, xmax, ymax])

            for i in range(len(W)):
                if W[i] == 2:
                    W[i] = 1

            label_list.append([C, W, gt_class_list, gt_boxes])

        return label_list

    def __getitem__(self, index):
        """
        :param index: a int generated by Dataloader in range [0, __len__()]
        :return: index-th element - containing all the aforementioned information
        """

        findex = self.index_list[index]     # findex refers to the file number
        fpath = os.path.join(self.img_dir, findex + '.jpg')

        img = Image.open(fpath)
        width, height = img.size
        #print(width, height, "w_h")
        img = transforms.ToTensor()(img)
        img = transforms.Resize((self.size, self.size))(img)
        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        lab_vec = self.anno_list[index][0]
        wgt_vec = self.anno_list[index][1]

        label = torch.FloatTensor(lab_vec)
        wgt = torch.FloatTensor(wgt_vec)

        gt_class_list, gt_boxes = self.anno_list[index][2], self.anno_list[index][3]

        """
        TODO:
            1. Load bounding box proposals for the index from self.roi_data. The proposals are of the format:
            [y_min, x_min, y_max, x_max] or [top left row, top left col, bottom right row, bottom right col]
            2. Normalize in the range (0, 1) according to image size (be careful of width/height and x/y correspondences)
            3. Make sure to return only the top_n proposals based on proposal confidence ("boxScores")!
            4. You may have to write a custom collate_fn since some of the attributes below may be variable in number for each data point -- okay I might need to do this
        """
        #print() # this should be arr -- there are 5011 images in the dataset
        proposals = self.roi_data["boxes"][:, index][0] # this seems more right, was [0, index] before
        #print(len(self.index_list), "index_len")
        #print(len(proposals))
        # it seems we divide everything by size
#         print(self.size, "image_size")
#         print(self.top_n, "top n")
#         print(self.roi_data["boxes"][0, index].shape)
#         print(self.roi_data["boxes"][0, index])
        # unpeel the arrs
        unwrapped_scores = []
        #print(self.roi_data["boxScores"][:, 0]) # (1, 5011); [0] (5011,); [:, 0] (1,) says it's a 1 array but is wrong
        # there seems to be 5011 roi props per image, far in excess of 300
        # the scores are presorted, no argsort needed, but you should do it any way
        #topn_scores = np.array(unwrapped_scores)[:self.top_n]
        #np.argsort(-1*np.array(unwrapped_scores))[:self.top_n]
        unwrapped_scores = self.roi_data["boxScores"][:, index]
        # for i in self.roi_data["boxScores"][:, index]:
        #     print(len(i), "scores") # yes these are the same len!!!!!!
        #     print(len(proposals), "proposals")
        #     raise Exception
        #     unwrapped_scores.append(i[0])
        # print(len(unwrapped_scores))
        
        #print(topn_scores)
        topn_proposals = []#unwrapped_scores[:, self.top_n]
        # what if there are less than n proposals!!
        # you should change [y_min, x_min, y_max, x_max] to (x1, y1, x2, y2) aka [1, 0, 3, 2]
        # minX, maxX, minY, maxY
        if len(proposals) > self.top_n:
            for i in range(self.top_n):
                topn_proposals.append(np.array([proposals[i][1]/width,  proposals[i][0]/height,
                                           proposals[i][3]/width, proposals[i][2]/height]))
        else: # need to pad here!
            for i in range(len(proposals)):
                topn_proposals.append(np.array([proposals[i][1]/width,  proposals[i][0]/height,
                                           proposals[i][3]/width, proposals[i][2]/height]))
            topn_proposals = topn_proposals + (self.top_n - len(proposals))*[[0, 0, 0, 0]] # this is right
            #print(topn_proposals, "top proposals")
            print("had to pad")
            #raise Exception # had tompad happened?
        # turn topn_proposisals to tensor
        topn_proposals = torch.tensor(np.array(topn_proposals)).float()
        ret = {}
        # to be save pad everything, but you shouldn't have to pad image and label
        # it has to see everything, don't collate here. 
        #print(gt_boxes, "gt_boxes_dataloader")
        ret['image'] = img
        ret['label'] = label
        ret['wgt'] = wgt
        ret['rois'] = topn_proposals
        ret['gt_boxes'] = torch.Tensor(gt_boxes)
        ret['gt_classes'] = gt_class_list
        return ret
    
# this should be debugged locally first
# it should return a dict for the whole batch
# rewrtie this
#  and figure out how to output the image and heatmap
def collate_fn(batch):
    # so when index a key, I want the tensor stack of that key for all the objects
    #print(batch)
    new_dict = dict()
    for datapoint in batch:
        for key, values in datapoint.items():
            #print(key)
            #print(type(values))
            #print(values.shape)
            #raise Exception
            if key in new_dict.keys():
                new_dict[key].append(values)
            else:
                new_dict[key] = [values]

    for key, values in new_dict.items():
        #print(key)
        #print(type(values))
        #print(values.shape)
        #raise Exception
        
        if type(values[0]) == torch.Tensor: # torch.stack
            new_dict[key] = torch.stack(values)
        #else:
            #new_dict[key] = values
    #print("poop")
    #print(new_dict)
    #print("image")
    #print(type(new_dict["image"]))
    #print(new_dict["image"].shape)
    #raise Exception
    #print("shloop")
    return new_dict

