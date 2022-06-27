import cv2.cv2 as cv2
import mkl_random
import numpy as np
import torch
from torch.utils.data import Dataset
import PIL.Image
import os.path
from pycocotools.coco import COCO

import scipy.misc
from tool import imutils
from torchvision import transforms
import random

import json





class COCOLabelData():

    def __init__(self, data=None, label=None):
        self.data = data
        self.label = label

class COCOLabelIdData():

    def __init__(self, data=None, label=None, id=None):
        self.data = data
        self.label = label
        self.id = id


id_to_index={
1 : 0 ,      # person
2 : 1 ,      # bicycle
3 : 2 ,      # car
4 : 3 ,      # motorcycle
5 : 4 ,      # airplane
6 : 5 ,      # bus
7 : 6 ,      # train
8 : 7 ,      # truck
9 : 8 ,      # boat
10 : 9 ,      # traffic light
11 : 10 ,      # fire hydrant
13 : 11 ,      # stop sign
14 : 12 ,      # parking meter
15 : 13 ,      # bench
16 : 14 ,      # bird
17 : 15 ,      # cat
18 : 16 ,      # dog
19 : 17 ,      # horse
20 : 18 ,      # sheep
21 : 19 ,      # cow
22 : 20 ,      # elephant
23 : 21 ,      # bear
24 : 22 ,      # zebra
25 : 23 ,      # giraffe
27 : 24 ,      # backpack
28 : 25 ,      # umbrella
31 : 26 ,      # handbag
32 : 27 ,      # tie
33 : 28 ,      # suitcase
34 : 29 ,      # frisbee
35 : 30 ,      # skis
36 : 31 ,      # snowboard
37 : 32 ,      # sports ball
38 : 33 ,      # kite
39 : 34 ,      # baseball bat
40 : 35 ,      # baseball glove
41 : 36 ,      # skateboard
42 : 37 ,      # surfboard
43 : 38 ,      # tennis racket
44 : 39 ,      # bottle
46 : 40 ,      # wine glass
47 : 41 ,      # cup
48 : 42 ,      # fork
49 : 43 ,      # knife
50 : 44 ,      # spoon
51 : 45 ,      # bowl
52 : 46 ,      # banana
53 : 47 ,      # apple
54 : 48 ,      # sandwich
55 : 49 ,      # orange
56 : 50 ,      # broccoli
57 : 51 ,      # carrot
58 : 52 ,      # hot dog
59 : 53 ,      # pizza
60 : 54 ,      # donut
61 : 55 ,      # cake
62 : 56 ,      # chair
63 : 57 ,      # couch
64 : 58 ,      # potted plant
65 : 59 ,      # bed
67 : 60 ,      # dining table
70 : 61 ,      # toilet
72 : 62 ,      # tv
73 : 63 ,      # laptop
74 : 64 ,      # mouse
75 : 65 ,      # remote
76 : 66 ,      # keyboard
77 : 67 ,      # cell phone
78 : 68 ,      # microwave
79 : 69 ,      # oven
80 : 70 ,      # toaster
81 : 71 ,      # sink
82 : 72 ,      # refrigerator
84 : 73 ,      # book
85 : 74 ,      # clock
86 : 75 ,      # vase
87 : 76 ,      # scissors
88 : 77 ,      # teddy bear
89 : 78 ,      # hair drier
90 : 79 ,      # toothbrush
}




class COCODataset(Dataset):

    def __init__(self, img_name_list_path, dataType, cocoRoot, transform=None):
        # self.img_name_list = load_img_name_list(img_name_list_path)
        # img_name_list_path = os.path.join(data_root, img_name_list_path)

        # dataType = "train2017"
        # annFile = os.path.join(cocoRoot, f'annotations_trainval2017/annotations/instances_{dataType}.json')
        # print(f'Annotation file: {annFile}')
        # coco = COCO(annFile)
        # ids = coco.getImgIds()
        # l = len(ids)
        # # classes_counter = np.zeros((80))
        # # for key in classes.keys():
        # #     print(key,'\t\t\t%d/%d\t %f%%'%(classes_counter[classes[key]],l,classes_counter[classes[key]]*100.0/l))
        # list = []
        # for i, f_id in enumerate(ids):
        #     # print()
        #     data = coco.loadImgs(f_id)[0]['file_name']
        #     anns = coco.getAnnIds(imgIds=f_id)
        #     anns = coco.loadAnns(anns)
        #     ret_label = np.zeros((80))
        #     for ann in anns:
        #         ret_label[id_to_index[ann['category_id']]] = 1
        #         # print(obj.label,end='\t\t')
        #         # if(obj.label in classes):
        #         #     ret_label[classes[obj.label]]=1
        #         # classes_counter[obj.label]+=1
        #         #     print(obj.label,'\t\t',' in the set')
        #         # else:
        #         #     print(obj.label,'\t\t', ' not in the set')
        #     # classes_counter += ret_label
        #     self.img_name_list.append(COCOLabelData(os.path.join(cocoRoot,dataType,), ret_label))
            # dict = {}
            # dict['data'] = data
            # dict['label'] = ret_label.tolist()
            # list.append(dict)
            # print(ret_label)
            # break

        if not os.path.isfile(img_name_list_path):
            print('Given json file not found: {}'.format(img_name_list_path))
            return
        with open(img_name_list_path, 'r') as f:
            jsonText = f.read()
            jsonObj = json.loads(jsonText)
            self.img_name_list = []
            for index,dect in enumerate(jsonObj['list']):
                self.img_name_list.append(COCOLabelIdData(dect['data'],np.array( dect['label']), dect['img']))

                # fromJsonText(jsonText)
        self.data_root = cocoRoot
        self.dataType = dataType
        self.transform = transform
        # print(self.img_name_list)

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        img_obj = self.img_name_list[idx]
        # print(img_obj)
        img = PIL.Image.open(os.path.join(self.data_root, self.dataType, img_obj.data)).convert("RGB")

        if self.transform:
            # print('transform')
            img = self.transform(img)

        return img_obj, img

class COCOClsDataset(COCODataset):

    def __init__(self, img_name_list_path, dataType, data_root, transform=None):
        super().__init__(img_name_list_path, dataType, data_root, transform)
        # self.label_list = load_image_label_list_from_npy(self.img_name_list)
        #self.label_list = load_image_label_list_from_xml(self.img_name_list, self.voc12_root)

    def __getitem__(self, idx):
        img_obj, img = super().__getitem__(idx)


        label = torch.from_numpy(img_obj.label)
        # print(img)
        # print(label)
        return img, label

class COCOClsDatasetMSF(COCODataset):

    def __init__(self, img_name_list_path, dataType, data_root,  inter_transform=None, unit=1):
        super().__init__(img_name_list_path, dataType, data_root, transform=None)
        # self.scales = scales
        self.unit = unit
        self.inter_transform = inter_transform

    def __getitem__(self, idx):
        img_obj, img = super().__getitem__(idx)


        img_orignal = img
        if self.inter_transform:
            img = self.inter_transform(img)


        # print(img_obj)
        return img_obj.id, img, torch.from_numpy(img_obj.label), np.array(img_orignal)

class COCOMaskData:
    def __init__(self, cam_path=None, mask_path=None, image_path=None, id=None):
        self.cam_path = cam_path
        self.mask_path = mask_path
        self.image_path = image_path
        self.id = id

class COCOMaskDataset(Dataset):

    def __init__(self, img_name_list_path, dataType, cocoRoot, transform_pic=None, transform_concat=None):


        if not os.path.isfile(img_name_list_path):
            print('Given json file not found: {}'.format(img_name_list_path))
            return
        # cocoRoot = "/home/wujiali/DeepLearning/dataset/coco-file"
        # dataType = "train2017"
        annFile = os.path.join(cocoRoot, f'annotations_trainval2017/annotations/instances_{dataType}.json')
        coco = COCO(annFile)
        with open(img_name_list_path, 'r') as f:
            jsonText = f.read()
            jsonObj = json.loads(jsonText)
            self.img_name_list = []
            for index,dect in enumerate(jsonObj['list']):
                # print(dect.keys())
                self.img_name_list.append(COCOMaskData(cam_path=dect['cam_path'],
                                                       mask_path=dect['mask_path'],
                                                       image_path=os.path.join(cocoRoot, dataType, coco.loadImgs(dect['image_id'])[0]['file_name']) ,
                                                       id=dect['image_id']))

        del coco

        random.shuffle(self.img_name_list)
        self.transform_pic = transform_pic
        self.transform_concat = transform_concat
        self.trans_mask = transforms.ToTensor()
        # self.transform = mask_root
        # self.transform = cam_root
        # print(self.img_name_list)

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        data_obj = self.img_name_list[idx]
        # print(img_obj)
        img = PIL.Image.open(data_obj.image_path).convert("RGB")
        cam = torch.from_numpy(np.load(data_obj.cam_path)).type(torch.float32)
        mask = self.trans_mask(cv2.imread(data_obj.mask_path, cv2.IMREAD_GRAYSCALE))
        # mask = cv2.imread()
        # print('max cam', torch.max(cam))
        if self.transform_pic:
            # print('transform')
            img = self.transform_pic(img)

        h, w = cam.size()

        concat = torch.concat((img, cam.view(1, h, w), mask.view(1, h, w)), dim=0)

        if self.transform_concat:
            concat = self.transform_concat(concat)
        # print('unique cam', torch.unique(concat[3, :, :]))
        return concat[0:4, :, :], concat[4, :, :]
