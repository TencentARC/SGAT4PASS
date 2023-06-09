"""Base segmentation dataset"""
import os
import random
import numpy as np
import torchvision

from PIL import Image, ImageOps, ImageFilter
from ...config import cfg
from IPython import embed
from segmentron.utils.equirect_rotation_fast import Rot_Equirect

__all__ = ['SegmentationDataset']


class SegmentationDataset(object):
    """Segmentation Base Dataset"""

    def __init__(self, root, split, mode, transform, base_size=520, crop_size=480):
        super(SegmentationDataset, self).__init__()
        self.root = os.path.join(cfg.ROOT_PATH, root)
        self.transform = transform
        self.split = split
        self.mode = mode if mode is not None else split
        self.base_size = base_size
        self.crop_size = self.to_tuple(crop_size)
        self.color_jitter = self._get_color_jitter()
        self.padding_length = None

    def to_tuple(self, size):
        if isinstance(size, (list, tuple)):
            return tuple(size)
        elif isinstance(size, (int, float)):
            return tuple((size, size))
        else:
            raise ValueError('Unsupport datatype: {}'.format(type(size)))

    def _get_color_jitter(self):
        color_jitter = cfg.AUG.COLOR_JITTER
        if color_jitter is None:
            return None
        if isinstance(color_jitter, (list, tuple)):
            # color jitter should be a 3-tuple/list if spec brightness/contrast/saturation
            # or 4 if also augmenting hue
            assert len(color_jitter) in (3, 4)
        else:
            # if it's a scalar, duplicate for brightness, contrast, and saturation, no hue
            color_jitter = (float(color_jitter),) * 3
        return torchvision.transforms.ColorJitter(*color_jitter)

    def _val_sync_transform(self, img, mask):
        outsize = self.crop_size
        short_size = min(outsize)
        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - outsize[1]) / 2.))
        y1 = int(round((h - outsize[0]) / 2.))
        img = img.crop((x1, y1, x1 + outsize[1], y1 + outsize[0]))
        mask = mask.crop((x1, y1, x1 + outsize[1], y1 + outsize[0]))

        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _img_transform(self, img):
        return np.array(img)

    def _mask_transform(self, mask):
        return np.array(mask).astype('int32')

    @property
    def num_class(self):
        """Number of categories."""
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        return 0
