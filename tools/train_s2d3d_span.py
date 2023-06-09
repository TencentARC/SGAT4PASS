import pandas as pd
from PIL import Image
from IPython import embed
from tabulate import tabulate
from torch.utils.tensorboard import SummaryWriter
from segmentron.config import cfg
from segmentron.utils.visualize import show_flops_params, get_color_pallete
from segmentron.utils.default_setup import default_setup
from segmentron.utils.options import parse_args
from segmentron.utils.filesystem import save_checkpoint
from segmentron.utils.score import SegmentationMetric
from segmentron.utils.distributed import *
from segmentron.solver.lr_scheduler import get_scheduler
from segmentron.solver.optimizer import get_optimizer
from segmentron.solver.loss import get_segmentation_loss
from segmentron.models.model_zoo import get_segmentation_model
from segmentron.data.dataloader import get_segmentation_dataset
from torchvision import transforms
import torch.nn.functional as F
import torch.utils.data as data
import torch.nn as nn
import torch
import logging
import time
import copy
import datetime
import os
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)


try:
    import apex
except:
    print('apex is not installed')


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.use_fp16 = cfg.TRAIN.APEX
        self.mIoU_log = {}
        self.writer = None
        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
        ])
        # dataset and dataloader
        data_kwargs = {'transform': input_transform,
                       'base_size': cfg.TRAIN.BASE_SIZE,
                       'crop_size': cfg.TRAIN.CROP_SIZE}

        data_kwargs_testval = {'transform': input_transform,
                               'base_size': cfg.TRAIN.BASE_SIZE,
                               'crop_size': cfg.TEST.CROP_SIZE}

        train_dataset = get_segmentation_dataset(
            cfg.DATASET.NAME, split=cfg.DATASET.TRAIN_SPLIT, fold=cfg.DATASET.FOLD, mode='train', **data_kwargs)
        test_dataset = get_segmentation_dataset(
            cfg.DATASET.NAME, split='val', fold=cfg.DATASET.FOLD, mode='val', **data_kwargs_testval)
        self.classes = test_dataset.classes

        # --- split epoch to iteration
        scale = 1  # split to epoch, 5 if pin
        self.iters_per_epoch = len(
            train_dataset) // (args.num_gpus * cfg.TRAIN.BATCH_SIZE)
        self.iters_per_epoch = self.iters_per_epoch // scale
        cfg.TRAIN.EPOCHS = cfg.TRAIN.EPOCHS * scale

        self.max_iters = cfg.TRAIN.EPOCHS * self.iters_per_epoch
        train_sampler = make_data_sampler(
            train_dataset, shuffle=True, distributed=args.distributed)
        train_batch_sampler = make_batch_data_sampler(
            train_sampler, cfg.TRAIN.BATCH_SIZE, self.max_iters, drop_last=True)

        test_sampler = make_data_sampler(test_dataset, False, args.distributed)
        test_batch_sampler = make_batch_data_sampler(
            test_sampler, cfg.TEST.BATCH_SIZE, drop_last=False)

        self.train_loader = data.DataLoader(dataset=train_dataset,
                                            batch_sampler=train_batch_sampler,
                                            num_workers=cfg.DATASET.WORKERS,
                                            pin_memory=True)
        self.test_loader = data.DataLoader(dataset=test_dataset,
                                           batch_sampler=test_batch_sampler,
                                           num_workers=cfg.DATASET.WORKERS,
                                           pin_memory=True)

        # create network
        self.model = get_segmentation_model().to(self.device)
        logging.info(self.model)

        # print params and flops
        if get_rank() == 0:
            try:
                show_flops_params(copy.deepcopy(self.model), args.device)
            except Exception as e:
                logging.warning('get flops and params error: {}'.format(e))
        if cfg.MODEL.BN_TYPE not in ['BN']:
            logging.info('Batch norm type is {}, convert_sync_batchnorm is not effective'.format(
                cfg.MODEL.BN_TYPE))
        elif args.distributed and cfg.TRAIN.SYNC_BATCH_NORM:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            logging.info('SyncBatchNorm is effective!')
        else:
            logging.info('Not use SyncBatchNorm!')
        # create criterion
        # assert cfg.DATASET.IGNORE_INDEX == 0
        if cfg.SOLVER.PER_PIXEL_WEIGHT is None:
            self.criterion = get_segmentation_loss(cfg.MODEL.MODEL_NAME, use_ohem=cfg.SOLVER.OHEM,
                                                   aux=cfg.SOLVER.AUX, aux_weight=cfg.SOLVER.AUX_WEIGHT,
                                                   ignore_index=cfg.DATASET.IGNORE_INDEX).to(self.device)
        else:
            self.criterion = get_segmentation_loss(cfg.MODEL.MODEL_NAME, use_ohem=cfg.SOLVER.OHEM,
                                                   aux=cfg.SOLVER.AUX, aux_weight=cfg.SOLVER.AUX_WEIGHT,
                                                   ignore_index=cfg.DATASET.IGNORE_INDEX,
                                                   reduction="none").to(self.device)

        # optimizer, for model just includes encoder, decoder(head and auxlayer).
        self.optimizer = get_optimizer(self.model)
        # apex
        if self.use_fp16:
            self.model, self.optimizer = apex.amp.initialize(
                self.model.cuda(), self.optimizer, opt_level="O1")
            logging.info('**** Initializing mixed precision done. ****')

        # lr scheduling
        self.lr_scheduler = get_scheduler(
            self.optimizer, max_iters=self.max_iters, iters_per_epoch=self.iters_per_epoch)
        # resume checkpoint if needed
        self.start_epoch = 0
        if args.resume and os.path.isfile(args.resume):
            name, ext = os.path.splitext(args.resume)
            assert ext == '.pkl' or '.pth', 'Sorry only .pth and .pkl files supported.'
            logging.info(
                'Resuming training, loading {}...'.format(args.resume))
            resume_sate = torch.load(args.resume, map_location=args.device)
            self.model.load_state_dict(resume_sate['state_dict'])
            self.start_epoch = resume_sate['epoch']
            logging.info(
                'resume train from epoch: {}'.format(self.start_epoch))
            if resume_sate['optimizer'] is not None and resume_sate['lr_scheduler'] is not None:
                logging.info(
                    'resume optimizer and lr scheduler from resume state..')
                self.optimizer.load_state_dict(resume_sate['optimizer'])
                self.lr_scheduler.load_state_dict(resume_sate['lr_scheduler'])

        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model,
                                                             device_ids=[
                                                                 args.local_rank],
                                                             output_device=args.local_rank,
                                                             find_unused_parameters=False)
        # evaluation metrics
        self.metric = SegmentationMetric(
            train_dataset.num_class, args.distributed)

        # --- monitor
        self.best_val_mIoU = 0.
        self.best_test_mIoU = 0.
        self.cur_val_mIoU = 0.
        self.cur_test_mIoU = 0.

    def train(self):
        self.save_to_disk = get_rank() == 0
        epochs, max_iters, iters_per_epoch = cfg.TRAIN.EPOCHS, self.max_iters, self.iters_per_epoch
        log_per_iters, val_per_iters = self.args.log_iter, self.args.val_epoch * \
            self.iters_per_epoch
        if self.save_to_disk:
            self.writer = SummaryWriter(cfg.TRAIN.MODEL_SAVE_DIR)
        start_time = time.time()
        logging.info('Start training, Total Epochs: {:d} = Total Iterations {:d}'.format(
            epochs, max_iters))

        self.model.train()
        iteration = self.start_epoch * iters_per_epoch if self.start_epoch > 0 else 0
        for (images, targets, loss_mask, _) in self.train_loader:
            epoch = iteration // iters_per_epoch + 1
            iteration += 1
            images = images.to(self.device)
            targets = targets.to(self.device)
            loss_mask = loss_mask.to(self.device)
            outputs, add_loss = self.model(images)
            loss_dict = self.criterion(outputs, targets)

            loss_dict['loss'] = torch.mul(loss_dict['loss'], loss_mask).mean()
            loss_dict.update(add_loss)

            losses = sum(loss for loss in loss_dict.values())
            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            self.optimizer.zero_grad()
            if self.use_fp16:
                with apex.amp.scale_loss(losses, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                losses.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            eta_seconds = ((time.time() - start_time) /
                           iteration) * (max_iters - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            if iteration % log_per_iters == 0 and self.save_to_disk:
                self.writer.add_scalar(
                    'train_total_loss', losses_reduced.item(), iteration)
                for k in loss_dict_reduced.keys():
                    self.writer.add_scalar(
                        k, loss_dict_reduced[k].item(), iteration)
                self.writer.add_scalar(
                    'lr', self.optimizer.param_groups[0]['lr'], iteration)
                logging.info(
                    "Epoch: {:d}/{:d} || Iters: {:d}/{:d} || Lr: {:.6f} || "
                    "Loss: {:.4f} || Cost Time: {} || Estimated Time: {}".format(
                        epoch, epochs, iteration % iters_per_epoch, iters_per_epoch,
                        self.optimizer.param_groups[0]['lr'], losses_reduced.item(
                        ),
                        str(datetime.timedelta(
                            seconds=int(time.time() - start_time))),
                        eta_string))

            if not self.args.skip_val and iteration % val_per_iters == 0:
                # self.validation(epoch)
                self.model.eval()
                self.test()

                if self.save_to_disk:
                    self.writer.add_scalar('mIoU', self.cur_test_mIoU, epoch)
                    self.mIoU_log[epoch] = self.cur_test_mIoU * 100
                    mIoU_df = pd.DataFrame(self.mIoU_log, index=[0])
                    mIoU_df.to_csv(os.path.join(
                        cfg.TRAIN.MODEL_SAVE_DIR, "mIoU.csv"))
                    save_checkpoint(self.args, self.model, epoch, self.optimizer,
                                    self.lr_scheduler, is_best=False, save_last=True)

                if self.cur_test_mIoU > self.best_test_mIoU:
                    self.best_test_mIoU = self.cur_test_mIoU
                    logging.info("Achieve Best_mIoU !!!")
                    save_checkpoint(self.args, self.model, epoch,
                                    self.optimizer, self.lr_scheduler, is_best=True)

                self.model.train()
        total_training_time = time.time() - start_time
        total_training_str = str(
            datetime.timedelta(seconds=total_training_time))
        logging.info("Total training time: {} ({:.4f}s / it)".format(total_training_str,
                                                                     total_training_time / max_iters))

    def test(self, vis=False):
        self.metric.reset()
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model
        torch.cuda.empty_cache()
        model.eval()
        logging.info("[test on target 1]")
        for i, (image, target, filename) in enumerate(self.test_loader):
            image = image.to(self.device)
            target = target.to(self.device)
            with torch.no_grad():
                output = model(image)[0]

            self.metric.update(output, target)

        synchronize()
        pixAcc, mIoU, category_iou = self.metric.get(return_category_iou=True)
        logging.info("[TEST END]  pixAcc: {:.3f}, mIoU: {:.3f}".format(
            pixAcc * 100, mIoU * 100))
        self.cur_test_mIoU = mIoU

        headers = ['class id', 'class name', 'iou']
        table = []
        for i, cls_name in enumerate(self.classes):
            table.append([cls_name, category_iou[i]])
        logging.info('Category iou: \n {}'.format(tabulate(
            table, headers, tablefmt='grid', showindex="always", numalign='center', stralign='center')))
        # torch.cuda.empty_cache()


if __name__ == '__main__':
    args = parse_args()
    # get config
    cfg.update_from_file(args.config_file)
    cfg.update_from_list(args.opts)
    cfg.PHASE = 'train' if not args.test else 'test'
    cfg.ROOT_PATH = root_path
    cfg.check_and_freeze()

    # setup python train environment, logger, seed..
    default_setup(args)

    # create a trainer and start train
    trainer = Trainer(args)
    if args.test:
        assert 'pth' in cfg.TEST.TEST_MODEL_PATH, 'please provide test model pth!'
        logging.info('test model......')
        trainer.test(args.vis)
    else:
        trainer.train()
        trainer.test()
