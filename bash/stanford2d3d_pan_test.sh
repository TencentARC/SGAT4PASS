# Traditional Testing
python -m torch.distributed.launch --nproc_per_node=2 --master_port 30005 tools/eval_s2d3d_span.py \
    --config-file configs/stanford2d3d_pan/SGAT4PASS_small_1080x1080_Fold_1.yaml \
    TEST.TEST_MODEL_PATH checkpoints/Stanford2D3D_Fold_1/best_model_fold_1.pth \
    DATASET.FOLD 1

python -m torch.distributed.launch --nproc_per_node=4 --master_port 30005 tools/eval_s2d3d_span.py \
    --config-file configs/stanford2d3d_pan/SGAT4PASS_small_1080x1080_Fold_2.yaml \
    TEST.TEST_MODEL_PATH checkpoints/Stanford2D3D_Fold_2/best_model_fold_2.pth \
    DATASET.FOLD 2

python -m torch.distributed.launch --nproc_per_node=4 --master_port 30005 tools/eval_s2d3d_span.py \
    --config-file configs/stanford2d3d_pan/SGAT4PASS_small_1080x1080_Fold_3.yaml \
    TEST.TEST_MODEL_PATH checkpoints/Stanford2D3D_Fold_3/best_model_fold_3.pth \
    DATASET.FOLD 3

# SGA Testing

python -m torch.distributed.launch --nproc_per_node=4 --master_port 30005 tools/eval_s2d3d_span.py \
    --config-file configs/stanford2d3d_pan/SGAT4PASS_small_1080x1080_Fold_1.yaml \
    TEST.TEST_MODEL_PATH checkpoints/Stanford2D3D_Fold_1/best_model_fold_1.pth \
    TEST.SGA True \
    DATASET.FOLD 1