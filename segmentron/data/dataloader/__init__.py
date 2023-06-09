"""
This module provides data loaders and transformers for popular vision datasets.
"""
from .stanford2d3d_pan_mask import Stanford2d3dPanMaskSegmentation

datasets = {
    'stanford2d3d_mask_pan': Stanford2d3dPanMaskSegmentation,
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)
