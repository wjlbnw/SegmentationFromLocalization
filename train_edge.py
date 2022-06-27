import numpy
import numpy as np
import torch
import random
import cv2.cv2 as cv2
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from data_edge.data_edge import COCOEdgeDataSet
from tool import pyutils, imutils, torchutils, visualization
import argparse
import importlib
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from network.refer_SEAM import Net
from network.edge_SEAM import DeepLabv3_plus
import voc12.data

from PIL import Image
from tool.worker_init_fns import worker_init_fn


def adaptive_min_pooling_loss(x):
    # This loss does not affect the highest performance, but change the optimial background score (alpha)
    n, c, h, w = x.size()
    k = h * w // 4
    x = torch.max(x, dim=1)[0]
    y = torch.topk(x.view(n, -1), k=k, dim=-1, largest=False)[0]
    y = F.relu(y, inplace=False)
    loss = torch.sum(y) / (k * n)
    return loss


def max_onehot(x):
    # n, c, h, w = x.size()
    x_max = torch.max(x, dim=1, keepdim=True)[0]
    x[x != x_max] = 0
    return x


class BinaryDiceLoss(torch.nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, input, targets):
        # 获取每个批次的大小 N
        N = targets.size()[0]

        # 平滑变量
        smooth = 1
        # 将宽高 reshape 到同一纬度
        input_flat = input.view(N, -1)
        targets_flat = targets.view(N, -1)

        # 计算交集
        intersection = input_flat * targets_flat
        N_dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
        # 计算一个批次中平均每张图的损失
        loss = 1 - torch.mean(N_dice_eff)

        return loss


def color_pro(pro):
    # H, W = pro.shape
    pro_255 = (pro * 255).astype(np.uint8)
    pro_255 = np.expand_dims(pro_255, axis=2)
    color = cv2.applyColorMap(pro_255, cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

    return color


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # dataset setting
    parser.add_argument("--cocoRoot", default='/home/wujiali/DeepLearning/dataset/coco-file', type=str)
    parser.add_argument("--dataType", default='train2017', type=str)

    
    # train setting
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--edge_batch_size", default=4, type=int)
    parser.add_argument("--max_epoches", default=8, type=int)
    # parser.add_argument("--per_train_iters", default=6000, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--edge_lr", default=0.001, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=2.5e-4, type=float)
    parser.add_argument("--crop_size", default=448, type=int)
    # model setting
    parser.add_argument("--categories", default='./coco_categories', type=str)
    parser.add_argument("--weights", default=None, type=str)
    parser.add_argument("--edge_weights", default=None, type=str)

    # save setting
    parser.add_argument("--session_name", default="toedge20220304", type=str)
    parser.add_argument("--tblog_dir", default='./train_in_coco', type=str)
    parser.add_argument("--ret_root", default='./train_ret', type=str)
    parser.add_argument("--out_root", default='/home/wujiali/DeepLearning/train_out/edge_out', type=str)

    args = parser.parse_args()

    # 初始化数据集目录
    # cocoRoot = args.cocoRoot
    # dataType = args.dataType

    # 初始化输出文件夹
    log_path = os.path.join(args.out_root, args.session_name, )
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    # 创建日志输出
    pyutils.Logger(os.path.join(log_path, args.session_name + '.log'))
    print(vars(args))

    tblogger = SummaryWriter(os.path.join(args.out_root, args.session_name, args.ret_root, args.tblog_dir))

    f = open(args.categories, 'r')
    categories = f.read()
    categories = categories.split('\n')
    print(categories)
    train_dataset = COCOEdgeDataSet(coco_root=args.cocoRoot, data_type=args.dataType, categories=categories,
                                    transform=transforms.Compose([
                                        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3,
                                                               hue=0.1),
                                        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                             std=(0.229, 0.224, 0.225))
                                    ]),
                                    transform_concat=transforms.Compose([
                                        transforms.RandomHorizontalFlip(),
                                        imutils.RandomResizeCrop(args.crop_size, min_scale=0.7, max_scale=1.3)
                                        # transforms.RandomResizedCrop(args.crop_size, scale=(0.8, 1.2),
                                        #                              interpolation=transforms.InterpolationMode.NEAREST)
                                    ]),
                                    )


    def worker_init_fn(worker_id):
        np.random.seed(1 + worker_id)


    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True,
                                   worker_init_fn=worker_init_fn)
    max_step = len(train_dataset) // args.batch_size * args.max_epoches

    
    ref_model = Net(train_dataset.class_counter)
    edge_model = DeepLabv3_plus(nInputChannels=4, n_classes=1, os=16, pretrained=True, freeze_bn=False, _print=True)

    ref_param_groups = ref_model.get_parameter_groups()
    ref_optimizer = torchutils.PolyOptimizer([
        {'params': ref_param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': ref_param_groups[1], 'lr': 2 * args.lr, 'weight_decay': 0},
        {'params': ref_param_groups[2], 'lr': 10 * args.lr, 'weight_decay': args.wt_dec},
        {'params': ref_param_groups[3], 'lr': 20 * args.lr, 'weight_decay': 0}
    ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)
    if args.weights:
        print('ref_model load weights...')
        weights_dict = torch.load(args.weights)
        # del weights_dict['fc8.weight']
        ref_model.load_state_dict(weights_dict)
        print('ref_model\'s weights loaded.')

    if args.edge_weights:
        print('edge_model load weights...')
        weights_dict = torch.load(args.edge_weights)
        # del weights_dict['last_conv.0.weight']
        edge_model.load_state_dict(weights_dict)
        print('edge_model\'s weights loaded.')

    ref_model = torch.nn.DataParallel(ref_model).cuda()
    ref_model.train()

    edge_optimizer = torch.optim.Adam(edge_model.parameters(), lr=args.edge_lr)
    edge_model = torch.nn.DataParallel(edge_model).cuda()
    edge_model.train()


    avg_meter = pyutils.AverageMeter('loss', 'loss_cls', 'loss_er', 'loss_ecr')
    dice_loss = BinaryDiceLoss()
    #
    timer = pyutils.Timer("Session started: ")
    edge_batch_size = args.edge_batch_size
    crop_size = args.crop_size
    edge_img = torch.zeros(size=(edge_batch_size, 3, crop_size, crop_size), dtype=torch.float32)
    edge_refer_map = torch.zeros(size=(edge_batch_size, 1, crop_size, crop_size), dtype=torch.float32)
    edge_refer_rv_map = torch.zeros(size=(edge_batch_size, 1, crop_size, crop_size), dtype=torch.float32)
    edge_seg = torch.zeros(size=(edge_batch_size, 1, crop_size, crop_size), dtype=torch.float32)
    edge_data_size = 0

    refer_iter = 0
    edge_iter = 0
    # print(train_dataset.class_counter)
    for ep in range(args.max_epoches):

        for iter, (img, label, seg) in enumerate(train_data_loader):

            scale_factor = 0.5
            img1 = img
            img2 = F.interpolate(img1, scale_factor=scale_factor, mode='bilinear',
                                 align_corners=True,
                                 recompute_scale_factor=False)
            N, C, H, W = img1.size()

            label = label.cuda(non_blocking=True).unsqueeze(2).unsqueeze(3)

            cam1, cam_rv1 = ref_model(img1)
            label1 = F.adaptive_avg_pool2d(cam1, (1, 1))
            loss_rvmin1 = adaptive_min_pooling_loss(cam_rv1 * label)
            refer_map = visualization.max_norm(cam1)
            refer_rv_map = visualization.max_norm(cam_rv1)
            cam1 = F.interpolate(refer_map, scale_factor=scale_factor, mode='bilinear',
                                 align_corners=True, recompute_scale_factor=False) * label
            cam_rv1 = F.interpolate(refer_rv_map, scale_factor=scale_factor, mode='bilinear',
                                    align_corners=True, recompute_scale_factor=False) * label

            cam2, cam_rv2 = ref_model(img2)
            label2 = F.adaptive_avg_pool2d(cam2, (1, 1))
            loss_rvmin2 = adaptive_min_pooling_loss(cam_rv2 * label)
            cam2 = visualization.max_norm(cam2) * label
            cam_rv2 = visualization.max_norm(cam_rv2) * label

            loss_cls1 = F.multilabel_soft_margin_loss(label1, label)
            loss_cls2 = F.multilabel_soft_margin_loss(label2, label)

            ns, cs, hs, ws = cam2.size()
            loss_er = torch.mean(torch.abs(cam1 - cam2))
            # cam1[:, 0, :, :] = 1 - torch.max(cam1[:, 1:, :, :], dim=1)[0]
            # cam2[:, 0, :, :] = 1 - torch.max(cam2[:, 1:, :, :], dim=1)[0]
            tensor_ecr1 = torch.abs(max_onehot(cam2.detach()) - cam_rv1)  # *eq_mask
            tensor_ecr2 = torch.abs(max_onehot(cam1.detach()) - cam_rv2)  # *eq_mask
            loss_ecr1 = torch.mean(torch.topk(tensor_ecr1.view(ns, -1), k=(int)(cs * hs * ws * 0.2), dim=-1)[0])
            loss_ecr2 = torch.mean(torch.topk(tensor_ecr2.view(ns, -1), k=(int)(cs * hs * ws * 0.2), dim=-1)[0])
            loss_ecr = loss_ecr1 + loss_ecr2

            loss_cls = (loss_cls1 + loss_cls2) / 2 + (loss_rvmin1 + loss_rvmin2) / 2
            loss = loss_cls + loss_er + loss_ecr

            ref_optimizer.zero_grad()
            loss.backward()
            ref_optimizer.step()
            refer_iter += 1
            avg_meter.add({'loss': loss.item(), 'loss_cls': loss_cls.item(), 'loss_er': loss_er.item(),
                           'loss_ecr': loss_ecr.item()})

            refer_map = refer_map.detach()
            refer_rv_map = refer_rv_map.detach()
            for b in range(N):
                for l in range(cs):
                    # print(label.size())
                    # print(b,l)
                    if label[b][l] == 1:
                        # print(edge_refer_map.size())
                        # print(refer_map.size())
                        edge_img[edge_data_size].copy_(img[b])
                        edge_refer_map[edge_data_size, 0].copy_(refer_map[b][l])
                        edge_refer_rv_map[edge_data_size, 0].copy_(refer_rv_map[b][l])
                        edge_seg[edge_data_size, 0].copy_(seg[b][l])
                        edge_data_size += 1
                        if edge_data_size >= edge_batch_size:
                            edge_data_size = 0
                            edge_seg = edge_seg.cuda(non_blocking=True)
                            edge_out1 = edge_model(edge_img, edge_refer_map)
                            # edge_out2 = edge_model(edge_img, edge_out1)
                            # edge_out3 = edge_model(edge_img, edge_out2)

                            edge_out_rv = edge_model(edge_img, edge_refer_rv_map)
                            # edge_out_rv = edge_model(edge_img, edge_out_rv)
                            # edge_out_rv = edge_model(edge_img, edge_out_rv)

                            edge_loss1 = dice_loss(edge_out1, edge_seg)
                            edge_loss_rv = dice_loss(edge_out_rv, edge_seg)
                            # edge_loss2 = dice_loss(edge_out2, edge_seg)
                            # edge_loss3 = dice_loss(edge_out3, edge_seg)
                            edge_loss_ds = dice_loss(edge_out1, edge_out_rv)

                            edge_loss = edge_loss1 + edge_loss_rv + edge_loss_ds

                            edge_optimizer.zero_grad()
                            edge_loss.backward()
                            edge_optimizer.step()
                            edge_iter += 1

                            if edge_iter % 100 == 0:
                                with torch.no_grad():
                                    print('Edge Iter:%5d/XXXXX' % (edge_iter),
                                          'loss:%.4f' % (edge_loss.item()),
                                          'lr: %.4f' % (edge_optimizer.param_groups[0]['lr']), flush=True)

                                    # Visualization for training process
                                    view_size = crop_size // 4
                                    img_8 = F.interpolate(edge_img, (view_size, view_size),mode='bilinear')[0].numpy().transpose((1, 2, 0))
                                    img_8 = np.ascontiguousarray(img_8)
                                    mean = (0.485, 0.456, 0.406)
                                    std = (0.229, 0.224, 0.225)
                                    img_8[:, :, 0] = (img_8[:, :, 0] * std[0] + mean[0]) * 255
                                    img_8[:, :, 1] = (img_8[:, :, 1] * std[1] + mean[1]) * 255
                                    img_8[:, :, 2] = (img_8[:, :, 2] * std[2] + mean[2]) * 255
                                    img_8[img_8 > 255] = 255
                                    img_8[img_8 < 0] = 0
                                    img_8 = img_8.astype(np.uint8)

                                    # input_img = img_8.transpose((2, 0, 1))
                                    refer =  F.interpolate(edge_refer_map, (view_size, view_size),mode='bilinear')[0][0].detach().cpu().numpy()
                                    seg_label = F.interpolate(edge_seg, (view_size, view_size),mode='bilinear')[0][0].detach().cpu().numpy()
                                    seg_out1 = F.interpolate(edge_out1, (view_size, view_size),mode='bilinear')[0][0].detach().cpu().numpy()
                                    # seg_out2 = F.interpolate(edge_out2, (view_size, view_size),mode='bilinear')[0][0].detach().cpu().numpy()
                                    # seg_out3 = F.interpolate(edge_out3, (view_size, view_size),mode='bilinear')[0][0].detach().cpu().numpy()

                                    refer_rv = F.interpolate(edge_refer_rv_map, (view_size, view_size),mode='bilinear')[0][0].detach().cpu().numpy()
                                    seg_rv = F.interpolate(edge_out_rv, (view_size, view_size),mode='bilinear')[0][0].detach().cpu().numpy()

                                    refer = color_pro(refer)
                                    refer_heat = cv2.addWeighted(img_8, 0.5, refer, 0.5, 0)
                                    refer_rv = color_pro(refer_rv)
                                    refer_rv_heat = cv2.addWeighted(img_8, 0.5, refer, 0.5, 0)

                                    seg_out1 = color_pro(seg_out1)
                                    seg_heat1 = cv2.addWeighted(img_8, 0.5, seg_out1, 0.5, 0)
                                    # seg_out2 = color_pro(seg_out2)
                                    # seg_heat2 = cv2.addWeighted(img_8, 0.5, seg_out2, 0.5, 0)
                                    # seg_out3 = color_pro(seg_out3)
                                    # seg_heat3 = cv2.addWeighted(img_8, 0.5, seg_out3, 0.5, 0)

                                    seg_rv = color_pro(seg_rv)
                                    seg_rv_heat = cv2.addWeighted(img_8, 0.5, seg_rv, 0.5, 0)

                                    seg_label = color_pro(seg_label)
                                    seg_label_heat = cv2.addWeighted(img_8, 0.5, seg_label, 0.5, 0)

                                    tblogger.add_scalars('edge_loss', {
                                        'edge_loss1': edge_loss1.item(),
                                        'edge_loss_rv': edge_loss_rv.item(),
                                        'edge_loss_ds': edge_loss_ds.item(),
                                        'edge_loss': edge_loss.item()
                                    }, edge_iter - 1)
                                    img_8 = img_8.transpose((2, 0, 1))
                                    img_8 = img_8.reshape((1, 3, view_size, view_size))
                                    tblogger.add_images('MASK_COMPARE',
                                                        numpy.concatenate(
                                                            (img_8,
                                                             seg_label.transpose((2, 0, 1)).reshape(1, 3, view_size, view_size),
                                                             refer.transpose((2, 0, 1)).reshape(1, 3, view_size, view_size),
                                                             refer_rv.transpose((2, 0, 1)).reshape(1, 3, view_size, view_size),
                                                             seg_out1.transpose((2, 0, 1)).reshape(1, 3, view_size, view_size),
                                                             # seg_out2.transpose((2, 0, 1)).reshape(1, 3, view_size, view_size),
                                                             # seg_out3.transpose((2, 0, 1)).reshape(1, 3, view_size, view_size),
                                                             seg_rv.transpose((2, 0, 1)).reshape(1, 3, view_size, view_size)
                                                             )), edge_iter - 1)
                                    tblogger.add_images('HEAT_COMPARE', numpy.concatenate(
                                        (img_8,
                                         seg_label_heat.transpose((2, 0, 1)).reshape(1, 3, view_size, view_size),
                                         refer_heat.transpose((2, 0, 1)).reshape(1, 3, view_size, view_size),
                                         refer_rv_heat.transpose((2, 0, 1)).reshape(1, 3, view_size, view_size),
                                         seg_heat1.transpose((2, 0, 1)).reshape(1, 3, view_size, view_size),
                                         # seg_heat2.transpose((2, 0, 1)).reshape(1, 3, view_size, view_size),
                                         # seg_heat3.transpose((2, 0, 1)).reshape(1, 3, view_size, view_size),
                                         seg_rv_heat.transpose((2, 0, 1)).reshape(1, 3, view_size, view_size))), edge_iter - 1)

                            if edge_iter % 5000 == 0:
                                torch.save(edge_model.module.state_dict(),
                                           os.path.join(args.out_root, args.session_name, args.ret_root,
                                                            args.session_name + '_edge_%06d.pth' % (edge_iter)))

            avg_meter.add({'loss': loss.item(), 'loss_cls': loss_cls.item(), 'loss_er': loss_er.item(),
                           'loss_ecr': loss_ecr.item()})

            if (ref_optimizer.global_step - 1) % 100 == 0:
                with torch.no_grad():
                    timer.update_progress(ref_optimizer.global_step / max_step)

                    print('Refer Iter:%5d/%5d' % (ref_optimizer.global_step - 1, max_step),
                          'loss:%.4f %.4f %.4f %.4f' % avg_meter.get('loss', 'loss_cls', 'loss_er', 'loss_ecr'),
                          'imps:%.1f' % ((iter + 1) * args.batch_size / timer.get_stage_elapsed()),
                          'Fin:%s' % (timer.str_est_finish()),
                          'lr: %.4f' % (ref_optimizer.param_groups[0]['lr']), flush=True)

                    avg_meter.pop()

                    # Visualization for training process
                    img_8 = img1[0].numpy().transpose((1, 2, 0))
                    img_8 = np.ascontiguousarray(img_8)
                    mean = (0.485, 0.456, 0.406)
                    std = (0.229, 0.224, 0.225)
                    img_8[:, :, 0] = (img_8[:, :, 0] * std[0] + mean[0]) * 255
                    img_8[:, :, 1] = (img_8[:, :, 1] * std[1] + mean[1]) * 255
                    img_8[:, :, 2] = (img_8[:, :, 2] * std[2] + mean[2]) * 255
                    img_8[img_8 > 255] = 255
                    img_8[img_8 < 0] = 0
                    img_8 = img_8.astype(np.uint8)

                    input_img = img_8.transpose((2, 0, 1))
                    h = H // 4
                    w = W // 4
                    p1 = F.interpolate(cam1, (h, w), mode='bilinear')[0].detach().cpu().numpy()
                    p2 = F.interpolate(cam2, (h, w), mode='bilinear')[0].detach().cpu().numpy()
                    p_rv1 = F.interpolate(cam_rv1, (h, w), mode='bilinear')[0].detach().cpu().numpy()
                    p_rv2 = F.interpolate(cam_rv2, (h, w), mode='bilinear')[0].detach().cpu().numpy()

                    image = cv2.resize(img_8, (w, h), interpolation=cv2.INTER_CUBIC).transpose((2, 0, 1))
                    CLS1, CAM1, _, _ = visualization.generate_vis(p1, None, image,
                                                                  func_label2color=visualization.VOClabel2colormap,
                                                                  threshold=None, norm=False)
                    CLS2, CAM2, _, _ = visualization.generate_vis(p2, None, image,
                                                                  func_label2color=visualization.VOClabel2colormap,
                                                                  threshold=None, norm=False)
                    CLS_RV1, CAM_RV1, _, _ = visualization.generate_vis(p_rv1, None, image,
                                                                        func_label2color=visualization.VOClabel2colormap,
                                                                        threshold=None, norm=False)
                    CLS_RV2, CAM_RV2, _, _ = visualization.generate_vis(p_rv2, None, image,
                                                                        func_label2color=visualization.VOClabel2colormap,
                                                                        threshold=None, norm=False)
                    # MASK = eq_mask[0].detach().cpu().numpy().astype(np.uint8)*255
                    loss_dict = {'loss': loss.item(),
                                 'loss_cls': loss_cls.item(),
                                 'loss_er': loss_er.item(),
                                 'loss_ecr': loss_ecr.item()}
                    # itr = optimizer.global_step - 1
                    tblogger.add_scalars('loss', loss_dict, refer_iter)
                    tblogger.add_scalar('lr', ref_optimizer.param_groups[0]['lr'], refer_iter)
                    tblogger.add_image('Image', input_img, refer_iter)
                    # tblogger.add_image('Mask', MASK, itr)
                    tblogger.add_image('CLS1', CLS1, refer_iter)
                    tblogger.add_image('CLS2', CLS2, refer_iter)
                    tblogger.add_image('CLS_RV1', CLS_RV1, refer_iter)
                    tblogger.add_image('CLS_RV2', CLS_RV2, refer_iter)
                    tblogger.add_images('CAM1', CAM1, refer_iter)
                    tblogger.add_images('CAM2', CAM2, refer_iter)
                    tblogger.add_images('CAM_RV1', CAM_RV1, refer_iter)
                    tblogger.add_images('CAM_RV2', CAM_RV2, refer_iter)
                    # break
            if refer_iter % 5000 == 0:
                torch.save(ref_model.module.state_dict(),
                           os.path.join(args.out_root, args.session_name, args.ret_root,
                                        args.session_name + '_refer_%06d.pth' % (refer_iter)))


        else:
            print('')
            timer.reset_stage()

    torch.save(ref_model.module.state_dict(), os.path.join(args.out_root, args.session_name, args.ret_root,
                                                           args.session_name + '_refer.pth'))
    torch.save(edge_model.module.state_dict(), os.path.join(args.out_root, args.session_name, args.ret_root,
                                                            args.session_name + '_edge.pth'))
