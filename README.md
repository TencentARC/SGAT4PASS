# SGAT4PASS

## Introduction
This is the official implementation of the paper SGAT4PASS: Spherical Geometry-Aware Transformer for PAnoramic Semantic Segmentation (IJCAI 2023).
![SGAT4PASS](figs/pipeline.png)

## Environments

```bash
conda create -n SGAT4PASS python=3.7.7
conda activate SGAT4PASS
cd ~/path/to/SGAT4PASS 
conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install -r requirements.txt
python setup.py develop --user
# Optional: install apex follow: https://github.com/NVIDIA/apex
```

## Data Preparation

Prepare datasets: 

- [Stanford2D3D](http://buildingparser.stanford.edu/dataset.html)

```
datasets/
├── Stanford2D3D
│   ├── area_1
│   ├── area_2
│   ├── area_3
│   ├── area_4
│   ├── area_5a
│   ├── area_5b
│   └── area_6
```
Prepare pretrained weights, which can be found in the public repository of [SegFormer](https://github.com/NVlabs/SegFormer).
```
pretrained/
├── mit_b1.pth
└── mit_b2.pth
```
## Train

For example, to use 4 A100 GPUs to run the experiments on Stanford2D3D dataset:

```bash

LR=0.00008
EPOCHS=150
PER_PIXEL_WEIGHT=0.3 # Panorama-Aware Loss Weight
OFFSET_WEIGHT=0.3 # SDPE Loss Weight
REPROJECTION=True # whether to use reprojection
X_MAX=10 # the maximum value of x-axis
Y_MAX=10 # the maximum value of y-axis
Z_MAX=360 # the maximum value of z-axis
FOLD=1 # the fold of the Stanford2D3D dataset

python -m torch.distributed.launch --nproc_per_node=4 --master_port 30005 tools/train_s2d3d_span.py \
    --config-file configs/stanford2d3d_pan/SGAT4PASS_small_1080x1080_line_axis_a100_xyz_mask_loss.yaml \
    TRAIN.MODEL_SAVE_DIR workdirs/SGAT4PASS_small_1080x1080_line_${OFFSET_WEIGHT}_axis_${OFFSET_WEIGHT}_mask_loss_${PER_PIXEL_WEIGHT}_lr${LR}_epoch_${EPOCHS} \
    SOLVER.LR $LR \
    TRAIN.EPOCHS $EPOCHS \
    SOLVER.PER_PIXEL_WEIGHT $PER_PIXEL_WEIGHT \
    SOLVER.OFFSET_LINE_WEIGHT $OFFSET_WEIGHT \
    SOLVER.OFFSET_AXIS_WEIGHT $OFFSET_WEIGHT \
    AUG.REPROJECTION $REPROJECTION \
    DATASET.FOLD $FOLD \
    AUG.X_MAX $X_MAX \
    AUG.Y_MAX $Y_MAX \
    AUG.Y_MAX $Z_MAX

```

## Test
Download the models from [GoogleDrive](https://drive.google.com/file/d/11MrFL6bThXFGIZEr_GE0HCivfBbZ3Dnv/view?usp=sharing) and put them in `./checkpoints/` folder.

```
./checkpoints
├── Stanford2D3D_Fold_1
│   ├── best_model_fold_1.pth
├── Stanford2D3D_Fold_2
│   ├── best_model_fold_2.pth
└── Stanford2D3D_Fold_3
    ├── best_model_fold_3.pth
```

### Traditional Testing

```bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port 30005 tools/eval_s2d3d_span.py \
    --config-file configs/stanford2d3d_pan/SGAT4PASS_small_1080x1080_Fold_1.yaml \
    TEST.TEST_MODEL_PATH checkpoints/Stanford2D3D_Fold_3/best_model_fold_1.pth \
    DATASET.FOLD 1

python -m torch.distributed.launch --nproc_per_node=4 --master_port 30005 tools/eval_s2d3d_span.py \
    --config-file configs/stanford2d3d_pan/SGAT4PASS_small_1080x1080_Fold_2.yaml \
    TEST.TEST_MODEL_PATH checkpoints/Stanford2D3D_Fold_3/best_model_fold_2.pth \
    DATASET.FOLD 2

python -m torch.distributed.launch --nproc_per_node=4 --master_port 30005 tools/eval_s2d3d_span.py \
    --config-file configs/stanford2d3d_pan/SGAT4PASS_small_1080x1080_Fold_3.yaml \
    TEST.TEST_MODEL_PATH checkpoints/Stanford2D3D_Fold_3/best_model_fold_3.pth \
    DATASET.FOLD 3

```

### SGA Testing
First, change the value of the `cfg.TEST.ROTATIONS` to the angle to be tested in the `segmentron/config/settings.py`.

Then, run bash as follows
```bash

python -m torch.distributed.launch --nproc_per_node=4 --master_port 30005 tools/eval_s2d3d_span.py \
    --config-file configs/stanford2d3d_pan/SGAT4PASS_small_1080x1080_Fold_1.yaml \
    TEST.TEST_MODEL_PATH checkpoints/Stanford2D3D_Fold_1/best_model_fold_1.pth \
    TEST.SGA True \
    DATASET.FOLD 1

```
Note that `TEST.SGA` must be set to True

## References
Our repository is heavily based on [Trans4PASS](https://github.com/jamycheung/Trans4PASS)

Thank them for their excellent work!

## License

This repository is under the Apache-2.0 license. For commercial use, please contact with the authors.


## Citations

If you are interested in this work, please cite the following works:

SGAT4PASS [[**PDF**](https://arxiv.org/pdf/2306.03403.pdf)]
```
@article{li2023sgat4pass,
      title={SGAT4PASS: Spherical Geometry-Aware Transformer for PAnoramic Semantic Segmentation}, 
      author={Xuewei Li and Tao Wu and Zhongang Qi and Gaoang Wang and Ying Shan and Xi Li},
      journal={arXiv preprint arXiv:2306.03403},
      year={2023}
}
```