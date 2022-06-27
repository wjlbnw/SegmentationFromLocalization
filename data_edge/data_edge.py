import cv2.cv2 as cv2
import mkl_random
import numpy as np
import torch
from torch.utils.data import Dataset
import PIL.Image
import os.path
from pycocotools.coco import COCO

from torchvision import transforms as transforms
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
id_map={
    'person': 1,
    'bicycle': 2,
    'car': 3,
    'motorcycle': 4,
    'airplane': 5,
    'bus':6,
    'train':7,
    'truck':8,
    'boat':9,
    'traffic light': 10,
    'fire hydrant': 11,

    'stop sign': 13,
    'parking meter': 14,
    'bench': 15,
    'bird': 16,
    'cat': 17,
    'dog': 18,
    'horse': 19,
    'sheep': 20,
    'cow': 21,
    'elephant': 22,
    'bear': 23,
    'zebra': 24,
    'giraffe': 25,

    'backpack': 27,
    'umbrella': 28,

    'handbag': 31,
    'tie': 32,
    'suitcase': 33,
    'frisbee': 34,
    'skis': 35,
    'snowboard': 36,
    'sports ball': 37,
    'kite': 38,
    'baseball bat': 39,
    'baseball glove': 40,
    'skateboard': 41,
    'surfboard': 42,
    'tennis racket': 43,
    'bottle': 44,

    'wine glass': 46,
    'cup': 47,
    'fork': 48,
    'knife': 49,
    'spoon': 50,
    'bowl': 51,
    'banana': 52,
    'apple': 53,
    'sandwich': 54,
    'orange': 55,
    'broccoli': 56,
    'carrot': 57,
    'hot dog': 58,
    'pizza': 59,
    'donut': 60,
    'cake': 61,
    'chair': 62,
    'couch': 63,
    'potted plant': 64,
    'bed': 65,

    'dining table': 67,

    'toilet': 70,

    'tv': 72,
    'laptop': 73,
    'mouse': 74,
    'remote': 75,
    'keyboard': 76,
    'cell phone': 77,
    'microwave': 78,
    'oven': 79,
    'toaster': 80,
    'sink': 81,
    'refrigerator': 82,

    'book': 84,
    'clock': 85,
    'vase': 86,
    'scissors': 87,
    'teddy bear': 88,
    'hair drier': 89,
    'toothbrush':90
}
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


class COCOEdgeDataSet(Dataset):
    def __init__(self, coco_root, data_type, categories, transform, transform_concat ):
        self.cocoRoot = coco_root
        self.dataType = data_type
        annFile = os.path.join(coco_root, f'annotations_trainval2017/annotations/instances_{data_type}.json')

        print(f'Annotation file: {annFile}')
        # 为实例注释初始化COCO的API
        self.coco = COCO(annFile)
        self.img_list = []


        self.index_map = dict()
        self.index_map[1] = 0  # person标签
        for index, category in enumerate(categories):
            self.index_map[id_map[category]] = index + 1
            # self.index_map[id_map[category]] = index
        self.cat_ids = self.index_map.keys()
        self.class_counter = len(self.cat_ids)
        ids = self.coco.getImgIds()
        label_sum = np.zeros((self.class_counter))
        for i, f_id in enumerate(ids):
            anns = self.coco.getAnnIds(imgIds=f_id)
            anns = self.coco.loadAnns(anns)
            flag = False
            label = np.zeros((self.class_counter))
            for ann in anns:
                # if ann['category_id'] in self.cat_ids:
                #     label[self.index_map[ann['category_id']]] = 1
                #     flag = True
                if ann['category_id'] == 1:
                    label[0] = 1
                elif ann['category_id'] in self.cat_ids:
                    label[self.index_map[ann['category_id']]] = 1
                    flag = True

            if flag:
                label_sum += label
                self.img_list.append(f_id)
        # 打印数据集子集信息
        print('==============================================================================')
        print('选择用于分类的categories：', categories)
        # print('person自动添加')
        l = len(self.img_list)
        print(f'共有{self.class_counter}类,包含图像{l}张')
        print('类别ID：', self.cat_ids)
        print('-------------------------------------------------------------------------')
        print('person', '\t\t\t%d/%d\t %f%%' % (
            label_sum[0], l, label_sum[0] * 100.0 / l))

        for cat in categories:
            print(cat, '\t\t\t%d/%d\t %f%%' % (
            label_sum[self.index_map[id_map[cat]]], l, label_sum[self.index_map[id_map[cat]]] * 100.0 / l))

        print('==============================================================================')
        # 图像数据转换
        self.toTensor = transforms.ToTensor()
        self.transform = transform
        self.transform_concat = transform_concat

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_id = self.img_list[idx]
        # print(img_obj)
        data = self.coco.loadImgs(img_id)[0]['file_name']
        img = PIL.Image.open(os.path.join(self.cocoRoot, self.dataType, data)).convert("RGB")
        img = self.toTensor(img)
        if self.transform:
            img = self.transform(img)
        _, h, w = img.size()
        seg = torch.zeros(size=(self.class_counter, h, w), dtype=torch.float32)
        anns = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(anns)
        # label = torch.zeros(self.class_counter)
        for ann in anns:
            if ann['category_id'] in self.cat_ids:
                mask = torch.from_numpy(self.coco.annToMask(ann))
                seg[self.index_map[ann['category_id']]].add_(mask)
                # print(torch.unique(mask))
                # print(torch.unique(seg))

        seg.clip(0, 1)
        concat = torch.concat((img, seg), dim=0)

        if self.transform_concat:
            concat = self.transform_concat(concat)
        seg = concat[3:3+self.class_counter, :, :]
        seg = seg.clip(0, 1.)
        # print(self.class_counter)
        # print(seg.size())
        label = torch.max(seg.view(self.class_counter, -1), dim=-1)[0].view(self.class_counter)

        return concat[0:3, :, :], label, seg

class COCOStuffLabelDataSet(Dataset):
    def __init__(self, coco_root, data_type, transform, transform_concat ):
        self.cocoRoot = coco_root
        self.dataType = data_type
        annFile = os.path.join(coco_root, f'stuff_annotations_trainval2017/annotations/stuff_{data_type}.json')

        print(f'Annotation file: {annFile}')
        # 为实例注释初始化COCO的API
        self.coco = COCO(annFile)
        self.img_list = []

        self.class_counter = 92
        ids = self.coco.getImgIds()
        label_sum = np.zeros((self.class_counter))
        for i, f_id in enumerate(ids):
            anns = self.coco.getAnnIds(imgIds=f_id)
            anns = self.coco.loadAnns(anns)
            label = np.zeros((self.class_counter))
            for ann in anns:
                label[ann['category_id'] - 92] = 1

            label_sum += label
            self.img_list.append(f_id)
        # 打印数据集子集信息
        print('==============================================================================')

        l = len(self.img_list)
        print(f'共有{self.class_counter}类,包含图像{l}张')
        print('-------------------------------------------------------------------------')
        print('person', '\t\t\t%d/%d\t %f%%' % (
            label_sum[0], l, label_sum[0] * 100.0 / l))

        for cat in self.coco.loadCats(range(92, 184)):
            print(cat, '\t\t\t%d/%d\t %f%%' % (
            label_sum[cat['id']-92], l, label_sum[cat['id']-92] * 100.0 / l))

        print('==============================================================================')
        # 图像数据转换
        self.toTensor = transforms.ToTensor()
        self.transform = transform
        self.transform_concat = transform_concat

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_id = self.img_list[idx]
        # print(img_obj)
        data = self.coco.loadImgs(img_id)[0]['file_name']
        img = PIL.Image.open(os.path.join(self.cocoRoot, self.dataType, data)).convert("RGB")
        img = self.toTensor(img)
        if self.transform:
            img = self.transform(img)
        _, h, w = img.size()
        seg = torch.zeros(size=(self.class_counter, h, w), dtype=torch.float32)
        anns = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(anns)
        # label = torch.zeros(self.class_counter)
        for ann in anns:
            mask = torch.from_numpy(self.coco.annToMask(ann))
            seg[ann['category_id']-92].add_(mask)


        seg.clip(0, 1)
        concat = torch.concat((img, seg), dim=0)

        if self.transform_concat:
            concat = self.transform_concat(concat)
        seg = concat[3:3+self.class_counter, :, :]
        seg.clip_(0, 1.)
        label = torch.max(seg.view(self.class_counter, -1), dim=-1)[0].view(self.class_counter)
        # print(label)
        return concat[0:3, :, :], label