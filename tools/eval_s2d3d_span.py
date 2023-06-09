from __future__ import print_function
from segmentron.utils.distributed import synchronize, make_data_sampler, make_batch_data_sampler, get_rank
from segmentron.utils.visualize import get_color_pallete
import pandas as pd
from segmentron.utils.equirect_rotation_fast import Rot_Equirect
from segmentron.utils.default_setup import default_setup
from segmentron.utils.options import parse_args
from segmentron.config import cfg
from segmentron.utils.score import SegmentationMetric
from segmentron.models.model_zoo import get_segmentation_model
from segmentron.data.dataloader import get_segmentation_dataset
from torchvision import transforms
from tabulate import tabulate
import torch.nn.functional as F
import torch.utils.data as data
import torch.nn as nn
import torch
import numpy as np
import logging

import os
import sys
from PIL import Image
cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)


NAME_CLASSES = [
    # 'unknown',
    'beam', 'board', 'bookcase', 'ceiling', 'chair',
    'clutter', 'column', 'door', 'floor', 'sofa',
    'table', 'wall', 'window']


def get_class_colors():
    return np.load('colors.npy')


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


miou_result = []
acc_result = []
per_class_miou = {}


class Evaluator(object):
    def __init__(self, args, rotation=None):
        self.args = args
        self.device = torch.device(args.device)
        self.eval_rotation = rotation
        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
        ])
        data_kwargs_testval = {'transform': input_transform,
                               'base_size': cfg.TRAIN.BASE_SIZE,
                               'crop_size': cfg.TEST.CROP_SIZE}
        if self.eval_rotation is not None:
            logging.info('Testing on rotation angle: x:{},y:{},z:{}'.format(
                self.eval_rotation[0], self.eval_rotation[1], self.eval_rotation[2]))
        val_dataset = get_segmentation_dataset('stanford2d3d_mask_pan', split='val', fold=cfg.DATASET.FOLD, mode='val',
                                               eval_rotation=self.eval_rotation, **data_kwargs_testval)
        val_sampler = make_data_sampler(val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(
            val_sampler, images_per_batch=cfg.TEST.BATCH_SIZE, drop_last=False)
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=cfg.DATASET.WORKERS,
                                          pin_memory=True)
        self.classes = val_dataset.classes
        # create network
        self.model = get_segmentation_model().to(self.device)

        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'named_modules') and \
                cfg.MODEL.BN_EPS_FOR_ENCODER:
            logging.info('set bn custom eps for bn in encoder: {}'.format(
                cfg.MODEL.BN_EPS_FOR_ENCODER))
            self.set_batch_norm_attr(
                self.model.encoder.named_modules(), 'eps', cfg.MODEL.BN_EPS_FOR_ENCODER)

        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[
                                                             args.local_rank], output_device=args.local_rank,
                                                             find_unused_parameters=True)
        self.model.to(self.device)

        self.metric = SegmentationMetric(
            val_dataset.num_class, args.distributed)

    def set_batch_norm_attr(self, named_modules, attr, value):
        for m in named_modules:
            if isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.SyncBatchNorm):
                setattr(m[1], attr, value)

    def eval(self):
        logging.info("Target eval.")
        self._eval(self.val_loader)

    def _eval(self, dataloader):
        self.metric.reset()
        self.model.eval()
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model

        logging.info(
            "Start validation, Total sample: {:d}".format(len(dataloader)))
        import time
        time_start = time.time()
        hist = np.zeros((len(NAME_CLASSES), len(NAME_CLASSES)))
        for i, (image, target, filename) in enumerate(dataloader):
            image = image.to(self.device)
            target = target.to(self.device)
            with torch.no_grad():
                output = model(image)[0]
            if args.save_path is not None:
                batchsize = image.shape[0]
                for i in range(batchsize):
                    pred = torch.argmax(output[i], 1).squeeze(0).cpu().numpy()
                    result_img = np.zeros(
                        (pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
                    pred_label = pred.astype(np.uint8)
                    class_colors = get_class_colors()
                    img = Image.open(filename[i]).convert('RGB').resize(
                        (pred_label.shape[1], pred_label.shape[0]))
                    if self.eval_rotation is not None and self.eval_rotation != (0, 0, 0):
                        img = Rot_Equirect(img, self.eval_rotation)
                        img = Image.fromarray(img)
                    img.save(
                        (os.path.join(args.save_path+'_source', os.path.basename(filename[i]))))
                    img = np.array(img)
                    label = target[i].cpu().numpy()
                    label_img = np.zeros(
                        (label.shape[0], label.shape[1], 3), dtype=np.uint8)
                    for x in range(pred_label.shape[0]):
                        for y in range(pred_label.shape[1]):
                            if pred_label[x][y] > 0:
                                result_img[x,
                                           y] = class_colors[pred_label[x][y]]
                            if label[x][y] > 0:
                                label_img[x, y] = class_colors[label[x][y]]
                    vis_image = result_img // 2 + img // 2
                    result_img = Image.fromarray(result_img)
                    result_img.save(os.path.join(
                        args.save_path+'_color', os.path.basename(filename[i])))
                    vis_image = Image.fromarray(np.uint8(vis_image))
                    vis_image.save(os.path.join(
                        args.save_path+'_vis', os.path.basename(filename[i])))
                    label_image = Image.fromarray(label_img)
                    label_image.save(os.path.join(
                        args.save_path+'_label', os.path.basename(filename[i])))
            self.metric.update(output, target)
            pixAcc, mIoU = self.metric.get()
            if i % 10 == 0:
                logging.info("Sample: {:d}, validation pixAcc: {:.3f}, mIoU: {:.3f}".format(
                    i + 1, pixAcc * 100, mIoU * 100))

        synchronize()
        pixAcc, mIoU, category_iou = self.metric.get(return_category_iou=True)
        logging.info('Eval use time: {:.3f} second'.format(
            time.time() - time_start))
        logging.info('End validation pixAcc: {:.3f}, mIoU: {:.3f}'.format(
            pixAcc * 100, mIoU * 100))
        miou_result.append(round(mIoU * 100, 3))
        acc_result.append(round(pixAcc * 100, 3))

        headers = ['class id', 'class name', 'iou']
        table = []
        for i, cls_name in enumerate(self.classes):
            if cls_name not in per_class_miou.keys():
                per_class_miou[cls_name] = []
            table.append([cls_name, category_iou[i]])
            per_class_miou[cls_name].append(category_iou[i])
        logging.info('Category iou: \n {}'.format(tabulate(table, headers, tablefmt='grid', showindex="always",
                                                           numalign='center', stralign='center')))


if __name__ == '__main__':
    args = parse_args()
    cfg.update_from_file(args.config_file)
    cfg.update_from_list(args.opts)
    cfg.PHASE = 'test'
    cfg.ROOT_PATH = root_path
    cfg.check_and_freeze()
    default_setup(args)
    if cfg.TEST.SGA is False:
        logging.info('Testing on original image')
        if args.save_path is not None and get_rank() == 0:
            if not os.path.exists(args.save_path+'_color'):
                os.mkdir(args.save_path+'_color')
            if not os.path.exists(args.save_path+'_vis'):
                os.mkdir(args.save_path+'_vis')
            if not os.path.exists(args.save_path+'_source'):
                os.mkdir(args.save_path+'_source')
            if not os.path.exists(args.save_path+'_label'):
                os.mkdir(args.save_path+'_label')
        evaluator = Evaluator(args)
        evaluator.eval()
    else:
        logging.info('Testing on rotation image')
        logging.info('Rotation angles: ' + str(cfg.TEST.ROTATIONS))
        if args.save_path is not None:
            base_name = args.save_path
        for angle in cfg.TEST.ROTATIONS:
            if args.save_path is not None and get_rank() == 0:
                args.save_path = base_name + \
                    str(angle[0]) + str(angle[1]) + str(angle[2])
                if not os.path.exists(args.save_path+'_color'):
                    os.mkdir(args.save_path+'_color')
                if not os.path.exists(args.save_path+'_vis'):
                    os.mkdir(args.save_path+'_vis')
                if not os.path.exists(args.save_path+'_source'):
                    os.mkdir(args.save_path+'_source')
                if not os.path.exists(args.save_path+'_label'):
                    os.mkdir(args.save_path+'_label')
            evaluator = Evaluator(args, rotation=angle)
            evaluator.eval()
        std_miou = np.std(miou_result)
        var_miou = np.var(miou_result)
        std_acc = np.std(acc_result)
        var_acc = np.var(acc_result)
        logging.info("std_miou: {}, var_miou: {}".format(std_miou, var_miou))
        logging.info(
            "std_per_pixel_acc: {}, var_per_pixel_acc: {}".format(std_acc, var_acc))
    logging.info("Test Over")
