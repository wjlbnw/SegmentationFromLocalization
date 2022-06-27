import numpy as np
import torch
import cv2
import os
import voc12.data
import scipy.misc
import importlib
from torch.utils.data import DataLoader
import torchvision
from tool import imutils, pyutils, visualization
import argparse
from PIL import Image
import torch.nn.functional as F
from network.deeplab_SEAM import Net
from network.edge_SEAM import DeepLabv3_plus
from torchvision import transforms
import pandas as pd
from tqdm import tqdm

rgb_map = [(0, 0, 0),  # 0
           (128, 0, 0),  # 0
           (0, 128, 0),  # 0
           (128, 128, 0),
           (0, 0, 128),
           (128, 0, 128),
           (0, 128, 128),
           (128, 128, 128),
           (64, 0, 0),
           (192, 0, 0),
           (64, 128, 0),
           (192, 128, 0),
           (64, 0, 128),
           (192, 0, 128),
           (64, 128, 128),
           (192, 128, 128),
           (0, 64, 0),
           (128, 64, 0),
           (0, 192, 0),
           (128, 192, 0),
           (0, 64, 128)]


def index2rgb(indexs):

    cmap = np.zeros((indexs.shape[0], indexs.shape[1], 3), np.uint8)
    for item_index in range(21):
        cmap[indexs == item_index, 0] = rgb_map[item_index][2]
        cmap[indexs == item_index, 1] = rgb_map[item_index][1]
        cmap[indexs == item_index, 2] = rgb_map[item_index][0]

    return cmap

categories = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
              'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
              'tvmonitor']
import numpy as np
from PIL import Image

# out = net.blobs['score'].data[0].argmax(axis=0)
def colorful(out, name):
    arr = out.astype(np.uint8)
    im = Image.fromarray(arr)

    palette = []
    for i in range(256):
        palette.extend((i, i, i))
    palette[:3 * 21] = np.array([[0, 0, 0],
                                 [128, 0, 0],
                                 [0, 128, 0],
                                 [128, 128, 0],
                                 [0, 0, 128],
                                 [128, 0, 128],
                                 [0, 128, 128],
                                 [128, 128, 128],
                                 [64, 0, 0],
                                 [192, 0, 0],
                                 [64, 128, 0],
                                 [192, 128, 0],
                                 [64, 0, 128],
                                 [192, 0, 128],
                                 [64, 128, 128],
                                 [192, 128, 128],
                                 [0, 64, 0],
                                 [128, 64, 0],
                                 [0, 192, 0],
                                 [128, 192, 0],
                                 [0, 64, 128]
                                 ], dtype='uint8').flatten()

    im.putpalette(palette)
    im.save(name)

class ResultCaluater:
    def __init__(self, gt_folder, num_cls, threshold):
        self.P = []
        self.T = []
        self.TP = []
        for i in range(num_cls):
            self.P.append(0)
            self.T.append(0)
            self.TP.append(0)



        self.gt_folder = gt_folder
        self.threshold = threshold
        self.num_cls = num_cls

    def append(self, tensor, img_name):
        tensor[0, :, :] = self.threshold
        predict = np.argmax(tensor, axis=0).astype(np.uint8)
        gt_file = os.path.join(self.gt_folder, '%s.png' % img_name)
        gt = np.array(Image.open(gt_file))
        cal = gt < 255
        mask = (predict == gt) * cal

        for i in range(self.num_cls):
            self.P[i] += np.sum((predict == i) * cal)
            self.T[i] += np.sum((gt == i) * cal)
            self.TP[i] += np.sum((gt == i) * mask)

    def caluate(self):
        IoU = []
        T_TP = []
        P_TP = []
        FP_ALL = []
        FN_ALL = []
        for i in range(self.num_cls):
            IoU.append(self.TP[i] / (self.T[i] + self.P[i] - self.TP[i] + 1e-10))
            T_TP.append(self.T[i] / (self.TP[i] + 1e-10))
            P_TP.append(self.P[i] / (self.TP[i] + 1e-10))
            FP_ALL.append((self.P[i] - self.TP[i]) / (self.T[i] + self.P[i] - self.TP[i] + 1e-10))
            FN_ALL.append((self.T[i] - self.TP[i]) / (self.T[i] + self.P[i] - self.TP[i] + 1e-10))
        loglist = {}
        for i in range(self.num_cls):
            loglist[categories[i]] = IoU[i] * 100

        miou = np.mean(np.array(IoU))
        loglist['mIoU'] = miou * 100

        for i in range(self.num_cls):
            if i % 2 != 1:
                print('%11s:%7.3f%%' % (categories[i], IoU[i] * 100), end='\t')
            else:
                print('%11s:%7.3f%%' % (categories[i], IoU[i] * 100))
        print('\n======================================================')
        print('%11s:%7.3f%%' % ('mIoU', miou * 100))
        return loglist

def writedict(file, dictionary):
    s = ''
    for key in dictionary.keys():
        sub = '%s:%s  ' % (key, dictionary[key])
        s += sub
    s += '\n'
    file.write(s)

def writelog(filepath, metric, comment):
    filepath = filepath
    logfile = open(filepath, 'a')
    import time
    logfile.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logfile.write('\t%s\n' % comment)
    writedict(logfile, metric)
    logfile.write('=====================================\n')
    logfile.close()


def writemUI(filepath, metric, comment):
    filepath = filepath
    logfile = open(filepath, 'a')
    logfile.write('\t%s\t%f\n' % (comment, metric['mIoU']))
    logfile.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--edge_weights", required=True, type=str)
    parser.add_argument("--infer_list", default="voc12/val.txt", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--voc12_root", default='VOC2012', type=str)
    parser.add_argument("--label_dir", default=None, type=str)
    parser.add_argument("--gt_dir", default='/home/wujiali/DeepLearning/dataset/VOC2012/SegmentationClass', type=str)
    parser.add_argument('--logfile', default='./evallog.txt', type=str)
    parser.add_argument('--mUIfile', default='./muilog.txt', type=str)
    # parser.add_argument("--out_crf", default=None, type=str)
    # parser.add_argument("--out_cam_pred", default=None, type=str)
    # parser.add_argument("--out_cam_pred_alpha", default=0.26, type=float)

    args = parser.parse_args()

    # outdir = args.out_cam
    # cam_dir = os.path.join(outdir, 'cam')
    # edge_dir = os.path.join(outdir, 'edge')
    # edge_rv_dir = os.path.join(outdir, 'edge_rv')
    def mkdir(path):
        if not os.path.exists(path):
            os.mkdir(path)

    # mkdir(args.label_dir)
    # mkdir(args.out_cam)
    # mkdir(cam_dir)
    # mkdir(edge_dir)
    # mkdir(edge_rv_dir)

    ref_model = Net()
    edge_model = DeepLabv3_plus(nInputChannels=4, n_classes=1, os=16, pretrained=True, freeze_bn=False, _print=False)

    ref_param_groups = ref_model.get_parameter_groups()

    print('ref_model load weights...')
    weights_dict = torch.load(args.weights)
    ref_model.load_state_dict(weights_dict)
    print('ref_model\'s weights loaded.')

    # print('edge_model load weights...')
    # weights_dict = torch.load(args.edge_weights)
    # edge_model.load_state_dict(weights_dict)
    # print('edge_model\'s weights loaded.')

    ref_model.eval()
    ref_model = ref_model.cuda()

    edge_model.eval()
    edge_model = edge_model.cuda()




    infer_dataset = voc12.data.VOC12ClsDataset(args.infer_list, voc12_root=args.voc12_root,
                                                  transform=torchvision.transforms.Compose(
                                                      [transforms.ToTensor(),
                                                       transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                            std=(0.229, 0.224, 0.225))
                                                       ]))

    infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    all_counter = len(infer_dataset)

    gt_folder = args.gt_dir
    with torch.no_grad():

        weight_name = args.edge_weights

        weights_dict = torch.load(os.path.join(args.edge_weights, weight_name))
        edge_model.load_state_dict(weights_dict)

        result1 = ResultCaluater(gt_folder, 21, 0.5)
        result2 = ResultCaluater(gt_folder, 21, 0.5)
        result3 = ResultCaluater(gt_folder, 21, 0.5)
        result_rv = ResultCaluater(gt_folder, 21, 0.5)

        for iter, (img_name, img, label) in tqdm(enumerate(infer_data_loader), total=len(infer_data_loader), dynamic_ncols=True):
            img_name = img_name[0]
            img = img.cuda()
            refer_map, cam_rv = ref_model(img)
            refer_map = visualization.max_norm(refer_map)
            refer_rv_map = visualization.max_norm(cam_rv)

            ns, cs, _, _ = refer_map.size()
            _, _, hs, ws = img.size()
            ret1 = torch.zeros(size=(cs, hs, ws), dtype=torch.float32)
            ret2 = torch.zeros(size=(cs, hs, ws), dtype=torch.float32)
            ret3 = torch.zeros(size=(cs, hs, ws), dtype=torch.float32)
            # ret_rv = torch.zeros(size=(cs, hs, ws), dtype=torch.float32)

            for i in range(1, cs):
                if label[0][i-1] == 1:


                    edge_out1 = edge_model(img, refer_map[:, i:i+1, :, :])
                    edge_out2 = edge_model(img, edge_out1)
                    edge_out3 = edge_model(img, edge_out2)
                    # edge_out_rv = edge_model(img, refer_rv_map[:, i:i+1, :, :])

                    ret1[i].copy_(edge_out1[0, 0, :, :].cpu())
                    ret2[i].copy_(edge_out2[0, 0, :, :].cpu())
                    ret3[i].copy_(edge_out3[0, 0, :, :].cpu())
                    # ret_rv[i].copy_(edge_out_rv[0, 0, :, :].cpu())


            result1.append(ret1.cpu().numpy(), img_name)
            result2.append(ret2.cpu().numpy(), img_name)
            result3.append(ret3.cpu().numpy(), img_name)

            # result_rv.append(ret_rv.cpu().numpy(), img_name)

            # if args.label_dir:
            #
            #     colorful(np.argmax(ret1.cpu().numpy(), axis=0).astype(np.uint8), os.path.join(args.label_dir, '%s.png' % img_name))
            #
            # if (iter + 1)%200 == 0:
            #     print('%4d-%4d-%s'%(iter+1, all_counter, weight_name))

        log1 = result1.caluate()
        log2 = result2.caluate()
        log3 = result3.caluate()
        log_rv = result_rv.caluate()
        # writelog(args.logfile, {'mIoU': log1}, 'out1_' + weight_name)
        # writelog(args.logfile, {'mIoU': log_rv}, 'out_rv' + weight_name)
        print('mIoU: %.3f%%\t %s' % (log1['mIoU'], 'out1_' + weight_name))
        print('mIoU: %.3f%%\t %s' % (log2['mIoU'], 'out2_' + weight_name))
        print('mIoU: %.3f%%\t %s' % (log3['mIoU'], 'out3_' + weight_name))
        print('mIoU: %.3f%%\t %s' % (log_rv['mIoU'], 'out_rv' + weight_name))
        # writelog(args.mUIfile, log1, 'out1_' + weight_name)
        # writelog(args.mUIfile, log_rv, 'out_rv' + weight_name)




