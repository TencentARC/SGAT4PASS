"""Stanford2D3D Panoramic Dataset."""
import glob
import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image, ImageOps, ImageFilter
from segmentron.data.dataloader.seg_data_base import SegmentationDataset
from segmentron.utils.equirect_rotation_fast import Rot_Equirect
from segmentron.config import cfg
import random

__FOLD__ = {
    '1_train': ['area_1', 'area_2', 'area_3', 'area_4', 'area_6'],
    '1_val': ['area_5a', 'area_5b'],
    '2_train': ['area_1', 'area_3', 'area_5a', 'area_5b', 'area_6'],
    '2_val': ['area_2', 'area_4'],
    '3_train': ['area_2', 'area_4', 'area_5a', 'area_5b'],
    '3_val': ['area_1', 'area_3', 'area_6'],
    'trainval': ['area_1', 'area_2', 'area_3', 'area_4', 'area_5a', 'area_5b', 'area_6'],
}


class Stanford2d3dPanMaskSegmentation(SegmentationDataset):
    """Stanford2d3d Semantic Segmentation Dataset."""
    NUM_CLASS = 13

    def __init__(self, root='datasets/Stanford2D3D', split='train', fold=1, mode=None,
                 transform=None, eval_rotation=None, **kwargs):
        super(Stanford2d3dPanMaskSegmentation, self).__init__(
            root, split, mode, transform, **kwargs)
        self.fold = fold
        assert os.path.exists(
            root), "Please put the data in {SEG_ROOT}/datasets/"
        self.images, self.masks = _get_stanford2d3d_pairs(
            root, self.fold, split)
        # self.crop_size = [2048, 1024]  # for inference only
        assert (len(self.images) == len(self.masks))
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in {}".format(
                os.path.join(root, split)))
        logging.info('Found {} images in the folder {}'.format(
            len(self.images), os.path.join(root, split)))
        with open('semantic_labels.json') as f:
            id2name = [name.split('_')[0] for name in json.load(f)] + ['<UNK>']
        with open('name2label.json') as f:
            name2id = json.load(f)
        self.colors = np.load('colors.npy')
        self.id2label = np.array([name2id[name] for name in id2name], np.uint8)
        self.eval_rotation = eval_rotation

    def _mask_transform(self, mask):
        return torch.LongTensor(np.array(mask).astype('int32'))

    def _sync_transform(self, img, mask, resize=False):
        img_w, img_h = img.size
        loss_weight = None
        # random mirror
        if cfg.AUG.MIRROR and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        # random_reprojection
        if cfg.AUG.REPROJECTION and random.random() <= 0.5:
            X = int(random.random() * cfg.AUG.X_MAX)
            Y = int(random.random() * cfg.AUG.Y_MAX)
            Z = int(random.random() * cfg.AUG.Z_MAX)
            img = Rot_Equirect(img, (X, Y, Z))
            mask = Rot_Equirect(mask, (X, Y, Z))
            img = Image.fromarray(np.uint8(img))
            mask = Image.fromarray(np.uint8(mask))

        if cfg.AUG.REPROJECT_PADDING:
            self.padding_length = int(img_w / 2)
            img = np.array(img, dtype=np.uint8)
            left_img = img[:, -self.padding_length:, :]
            right_img = img[:, :self.padding_length, :]
            img = np.concatenate([left_img, img, right_img], axis=1)

            mask = np.array(mask, dtype=np.uint8)
            left_mask = mask[:, :-self.padding_length]
            right_mask = mask[:, :self.padding_length]
            mask = np.concatenate([left_mask, mask, right_mask], axis=1)

            img = Image.fromarray(img)
            mask = Image.fromarray(mask)
        # random scale new
        if cfg.AUG.RESCALE:
            short_size = random.randint(
                int(self.base_size * 0.5), int(self.base_size * 2.0))
            w, h = img.size
            oh = short_size
            ow = int(1.0 * w * oh / h)
            scale = 1.0 * oh / h
            ow = int(1.0 * w * scale)
            if cfg.AUG.REPROJECT_PADDING:
                self.padding_length = int(1.0 * self.padding_length * scale)

            img = img.resize((ow, oh), Image.BILINEAR)
            mask = mask.resize((ow, oh), Image.NEAREST)
        # random crop
        if cfg.AUG.CROP:
            crop_size = self.crop_size
            # pad crop
            if short_size < min(crop_size):
                padh = crop_size[0] - oh if oh < crop_size[0] else 0
                padw = crop_size[1] - ow if ow < crop_size[1] else 0
                img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
                mask = ImageOps.expand(
                    mask, border=(0, 0, padw, padh), fill=-1)
            if cfg.SOLVER.PER_PIXEL_WEIGHT is not None:
                nw, nh = img.size
                loss_weight = np.zeros((nh, nw), dtype=float)
                if nh % 2 == 0:
                    a = np.linspace(0, 90, num=int(nh/2))
                    b = np.concatenate((np.flip(a), a))
                    loss_weight += np.reshape(b, (nh, 1))
                    loss_weight = np.cos(np.deg2rad(
                        loss_weight)) * cfg.SOLVER.PER_PIXEL_WEIGHT + 1
                else:
                    a = np.linspace(0, 90, num=int(nh/2) + 1)
                    b = np.concatenate((np.flip(a), a[1:]))
                    loss_weight += np.reshape(b, (nh, 1))
                    loss_weight = np.cos(np.deg2rad(
                        loss_weight)) * cfg.SOLVER.PER_PIXEL_WEIGHT + 1

            # random crop crop_size
            w, h = img.size
            if cfg.AUG.REPROJECT_PADDING:
                start_w = max(int(self.padding_length - crop_size[1]/2), 0)
                end_w = min(int(w - self.padding_length -
                            crop_size[1]/2), w - crop_size[0])
                if h >= crop_size[0]:
                    y1 = random.randint(0, h - crop_size[0])

                x1 = random.randint(start_w, end_w)
            else:
                x1 = random.randint(0, w - crop_size[1])
                y1 = random.randint(0, h - crop_size[0])
            img = img.crop((x1, y1, x1 + crop_size[1], y1 + crop_size[0]))
            mask = mask.crop((x1, y1, x1 + crop_size[1], y1 + crop_size[0]))
            if loss_weight is not None:
                loss_weight = loss_weight[y1:y1 +
                                          crop_size[0], x1:x1 + crop_size[1]]
        # --- random perspective
        if cfg.AUG.PERSPECTIVE:
            p = torchvision.transforms.RandomPerspective(
                distortion_scale=0.8, p=1, fill=-1)
            img = p(img)
            mask = p(mask)

        # gaussian blur as in PSP
        if cfg.AUG.BLUR_PROB > 0 and random.random() < cfg.AUG.BLUR_PROB:
            radius = cfg.AUG.BLUR_RADIUS if cfg.AUG.BLUR_RADIUS > 0 else random.random()
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))

        # color jitter
        if self.color_jitter and random.random() < 0.5:
            img = self.color_jitter(img)
        # first resize image to fix size
        if resize:
            img = img.resize(
                (self.crop_size[1], self.crop_size[0]), Image.BILINEAR)
            mask = mask.resize(
                (self.crop_size[1], self.crop_size[0]), Image.NEAREST)
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        if cfg.SOLVER.PER_PIXEL_WEIGHT is not None and cfg.SOLVER.PER_PIXEL_NORM is True:
            loss_weight = loss_weight / np.mean(loss_weight)
        loss_weight = torch.Tensor(loss_weight)
        return img, mask, loss_weight

    def _val_sync_transform_resize(self, img, mask):
        # self.crop_size = [2048, 1024]  # for inference only
        short_size = self.crop_size  # for inference only
        if self.eval_rotation is not None and self.eval_rotation != (0, 0, 0):
            img = Rot_Equirect(img, self.eval_rotation)
            mask = Rot_Equirect(mask, self.eval_rotation)
            img = Image.fromarray(np.uint8(img))
            mask = Image.fromarray(np.uint8(mask))
        img = img.resize(short_size, Image.BICUBIC)
        mask = mask.resize(short_size, Image.NEAREST)

        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            img = self._img_transform(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        mask = Image.open(self.masks[index])
        mask = _color2id(mask, img, self.id2label)
        if self.mode == 'train':
            img, mask, loss_weight = self._sync_transform(
                img, mask, resize=True)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform_resize(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._val_sync_transform_resize(img, mask)

        if self.transform is not None:
            img = self.transform(img)
        mask[mask == 255] = -1
        if self.mode == 'train':
            return img, mask, loss_weight, self.images[index].split(self.root+'/')[-1]
        else:
            return img, mask, self.images[index].split(self.root+'/')[-1]

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 1

    @property
    def classes(self):
        """Category names.'unknown', """
        return ('beam', 'board', 'bookcase', 'ceiling', 'chair',
                'clutter', 'column', 'door', 'floor', 'sofa',
                'table', 'wall', 'window')


def _get_stanford2d3d_pairs(folder, fold, mode='train'):
    '''image is jpg, label is png'''
    img_paths = []
    if mode == 'train':
        area_ids = __FOLD__['{}_{}'.format(fold, mode)]
    elif mode == 'val':
        area_ids = __FOLD__['{}_{}'.format(fold, mode)]
    elif mode == 'trainval':
        area_ids = __FOLD__[mode]
    else:
        raise NotImplementedError
    for a in area_ids:
        img_paths += glob.glob(os.path.join(folder,
                               '{}/pano/rgb/*_rgb.png'.format(a)))
    img_paths = sorted(img_paths)
    mask_paths = [imgpath.replace('rgb', 'semantic') for imgpath in img_paths]
    return img_paths, mask_paths


def _color2id(mask, img, id2label):
    mask = np.array(mask, np.int32)
    rgb = np.array(img, np.int32)
    unk = (mask[..., 0] != 0)
    mask = id2label[mask[..., 1] * 256 + mask[..., 2]]
    mask[unk] = 0
    mask[rgb.sum(-1) == 0] = 0
    mask -= 1  # 0->255
    return Image.fromarray(mask)


if __name__ == '__main__':
    from torchvision import transforms
    import torch.utils.data as data
    # Transforms for Normalization
    input_transform = transforms.Compose([transforms.ToTensor(
    ), transforms.Normalize((.485, .456, .406), (.229, .224, .225)),])
    # Create Dataset
    trainset = Stanford2d3dPanMaskSegmentation(
        split='train', transform=input_transform)
    # Create Training Loader
    train_data = data.DataLoader(trainset, 4, shuffle=True, num_workers=0)
    for i, data in enumerate(train_data):
        imgs, targets, _ = data
        print(imgs.shape)
