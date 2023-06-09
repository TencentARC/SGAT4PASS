LR=0.00008
EPOCHS=150
PER_PIXEL_WEIGHT=0.3
OFFSET_WEIGHT=0.3
REPROJECTION=True
X_MAX=10
Y_MAX=10
Z_MAX=360
FOLD=1

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

