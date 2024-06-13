#!/bin/sh

## Random Baseline
python main.py --mode eval \
    --task_name sdqes \
    --classification_layer_name cosine_similarity \
    --model_name random \
    --data_path /vision/u/eatang/sdas/ \
    --video_path /scr/ego4d_full_frames/ \
    --backbone_name egovlp_base \
    --seed 8 \
    --num_workers 16 \
    --wandb_group latest \
    --gpus 1 \
    --batch_size 8 \
    --n_frames 60 \
    --frame_sample_rate 1 \
    --temporal_pooling_name identity \
    --criterion_name bce \
    --load_from rgb \
    --precision 16 \
    --n_frames_extra_val 300 \
    --batch_size_extra_val 2 


### Long video evals
# STREAM_VAL=1 python main.py --mode predict \
#     --task_name sdqes \
#     --classification_layer_name cosine_similarity \
#     --data_path /vision/u/eatang/sdas/ \
#     --video_path /scr/ego4d_full_frames/ \
#     --backbone_name qrnn_adapter_egovlp_base \
#     --seed 42 \
#     --num_workers 16 \
#     --gpus 1 \
#     --batch_size 1 \
#     --n_frames 60 \
#     --frame_sample_rate 1 \
#     --temporal_pooling_name identity \
#     --criterion_name bce \
#     --load_from rgb \
#     --precision 16 \
#     --backbone_clip_length 1 \
#     --backbone_unfreeze_layer_norm False \
#     --adapter_upsample_zero_init True \
#     --backbone_qrnn_bidirectional False \
#     --backbone_drop_path_rate 0.3 \
#     --backbone_proj_after True \
#     --num_qrnn_adapters 2 \
#     --temporal_pool_backbone False \
#     --qrnn_lookback 1 \
#     --qrnn_lookahead 0 \
#     --adapter_downsample_ratio 0.5 \
#     --checkpoint_path /vision/u/eatang/sdas/sdqes/5crftn7q/checkpoints/ckpt-epoch-epoch=49.ckpt \
#     --downsample_qrnn_adapter False \
#     --vanilla_adapter True


## QRNN Adapter full video
# STREAM_VAL=1 python main.py --mode predict \
#     --task_name sdqes \
#     --classification_layer_name cosine_similarity \
#     --data_path /vision/u/eatang/sdas/ \
#     --video_path /scr/ego4d_full_frames/ \
#     --backbone_name qrnn_adapter_clip_ViT-B/16 \
#     --seed 42 \
#     --num_workers 16 \
#     --gpus 1 \
#     --batch_size 1 \
#     --n_frames 60 \
#     --frame_sample_rate 1 \
#     --temporal_pooling_name identity \
#     --criterion_name bce \
#     --load_from rgb \
#     --precision 16 \
#     --backbone_unfreeze_layer_norm False \
#     --adapter_upsample_zero_init True \
#     --backbone_qrnn_bidirectional False \
#     --backbone_drop_path_rate 0.3 \
#     --backbone_proj_after True \
#     --num_qrnn_adapters 2 \
#     --temporal_pool_backbone False \
#     --qrnn_lookback 1 \
#     --qrnn_lookahead 0 \
#     --adapter_downsample_ratio 0.5 \
#     --downsample_qrnn_adapter False \
#     --vanilla_adapter True \
#     --checkpoint_path /vision/u/eatang/sdas/sdqes/jso7gkce/checkpoints/ckpt-epoch-epoch=49.ckpt

# LAVILA EVAL
# python main.py --mode eval \
#     --task_name sdqes \
#     --classification_layer_name cosine_similarity \
#     --data_path /vision/u/eatang/sdas/ \
#     --video_path /scr/ego4d_full_frames/ \
#     --backbone_name egovlp_base \
#     --seed 42 \
#     --num_workers 16 \
#     --wandb_group zs_egovlp_cl_1 \
#     --gpus 1 \
#     --batch_size 8 \
#     --n_frames 60 \
#     --frame_sample_rate 1 \
#     --temporal_pooling_name identity \
#     --criterion_name bce \
#     --load_from rgb \
#     --precision 16 \
#     --backbone_clip_length 1 \
#     --n_frames_extra_val 300 \
#     --batch_size_extra_val 2 

## LAVILA Adapter eval
# python main.py --mode eval \
#     --task_name sdqes \
#     --classification_layer_name cosine_similarity \
#     --data_path /vision/u/eatang/sdas/ \
#     --video_path /scr/ego4d_full_frames/ \
#     --backbone_name qrnn_adapter_lavila_base \
#     --seed 42 \
#     --num_workers 16 \
#     --gpus 1 \
#     --batch_size 2 \
#     --n_frames 300 \
#     --frame_sample_rate 1 \
#     --temporal_pooling_name identity \
#     --criterion_name bce \
#     --load_from rgb \
#     --precision 16 \
#     --backbone_clip_length 1 \
#     --backbone_unfreeze_layer_norm False \
#     --adapter_upsample_zero_init True \
#     --backbone_qrnn_bidirectional False \
#     --backbone_drop_path_rate 0.3 \
#     --backbone_proj_after True \
#     --num_qrnn_adapters 2 \
#     --temporal_pool_backbone False \
#     --qrnn_lookback 1 \
#     --qrnn_lookahead 0 \
#     --adapter_downsample_ratio 0.5 \
#     --checkpoint_path /vision/u/eatang/sdas/sdqes/mc9u6v8w/checkpoints/ckpt-epoch-epoch=49.ckpt \
#     --downsample_qrnn_adapter False \
#     --vanilla_adapter True \
#     --toy_dataloader True

## EGOVLP EVAL
# python main.py --mode eval \
#     --task_name sdqes \
#     --classification_layer_name cosine_similarity \
#     --data_path /vision/u/eatang/sdas/ \
#     --video_path /scr/ego4d_full_frames/ \
#     --backbone_name qrnn_adapter_egovlp_base \
#     --seed 42 \
#     --num_workers 16 \
#     --gpus 1 \
#     --batch_size 8 \
#     --n_frames 300 \
#     --frame_sample_rate 1 \
#     --temporal_pooling_name identity \
#     --criterion_name bce \
#     --load_from rgb \
#     --precision 16 \
#     --backbone_clip_length 1 \
#     --backbone_unfreeze_layer_norm False \
#     --adapter_upsample_zero_init True \
#     --backbone_qrnn_bidirectional False \
#     --backbone_drop_path_rate 0.3 \
#     --backbone_proj_after True \
#     --num_qrnn_adapters 2 \
#     --temporal_pool_backbone False \
#     --qrnn_lookback 1 \
#     --qrnn_lookahead 0 \
#     --adapter_downsample_ratio 0.323 \
#     --downsample_qrnn_adapter False \
#     --retnet_adapter True \
#     --checkpoint_path /vision/u/eatang/sdas/sdqes/bgnxwg50/checkpoints/ckpt-epoch-epoch=49.ckpt



### CLIP EVAL
# python main.py --mode eval \
#     --task_name sdqes \
#     --classification_layer_name cosine_similarity \
#     --data_path /vision/u/eatang/sdas/ \
#     --video_path /scr/ego4d_full_frames/ \
#     --backbone_name clip_ViT-B/16 \
#     --seed 42 \
#     --num_workers 16 \
#     --gpus 1 \
#     --batch_size 8 \
#     --n_frames 60 \
#     --frame_sample_rate 1 \
#     --temporal_pooling_name identity \
#     --criterion_name bce \
#     --load_from rgb \
#     --precision 16 \
#     --n_frames_extra_val 300 \
#     --batch_size_extra_val 2
    # --checkpoint_path /vision/u/eatang/sdas/sdqes/i27ddbul/checkpoints/ckpt-epoch-epoch=49.ckpt \



#### Vanilla Adapter Eval
# python main.py --mode eval \
#     --task_name sdqes \
#     --classification_layer_name cosine_similarity \
#     --data_path /vision/u/eatang/sdas/ \
#     --video_path /scr/ego4d_full_frames/ \
#     --backbone_name qrnn_adapter_clip_ViT-B/16 \
#     --checkpoint_path /vision/u/eatang/sdas/sdqes/3to3z3da/checkpoints/ckpt-epoch-epoch=49.ckpt \
#     --seed 42 \
#     --num_workers 16 \
#     --gpus 1 \
#     --wandb_group eval \
#     --batch_size 8 \
#     --n_frames 60 \
#     --frame_sample_rate 1 \
#     --temporal_pooling_name identity \
#     --criterion_name bce \
#     --load_from rgb \
#     --precision 16 \
#     --adapter_upsample_zero_init True \
#     --backbone_qrnn_bidirectional False \
#     --backbone_drop_path_rate 0.3 \
#     --backbone_proj_after True \
#     --num_qrnn_adapters 2 \
#     --temporal_pool_backbone False \
#     --adapter_downsample_ratio 0.5 \
#     --downsample_qrnn_adapter False \
#     --vanilla_adapter True

### QRNN Adapter EVAL
# python main.py --mode eval \
#     --task_name sdqes \
#     --classification_layer_name cosine_similarity \
#     --data_path /vision/u/eatang/sdas/ \
#     --video_path /scr/ego4d_full_frames/ \
#     --backbone_name qrnn_adapter_clip_ViT-B/16 \
#     --seed 42 \
#     --num_workers 16 \
#     --gpus 1 \
#     --batch_size 8 \
#     --n_frames 60 \
#     --frame_sample_rate 1 \
#     --temporal_pooling_name identity \
#     --criterion_name bce \
#     --load_from rgb \
#     --precision 16 \
#     --backbone_unfreeze_layer_norm False \
#     --adapter_upsample_zero_init True \
#     --backbone_qrnn_bidirectional False \
#     --backbone_drop_path_rate 0.3 \
#     --backbone_proj_after True \
#     --num_qrnn_adapters 2 \
#     --temporal_pool_backbone False \
#     --qrnn_lookback 1 \
#     --qrnn_lookahead 0 \
#     --adapter_downsample_ratio 0.35 \
#     --adapter_upsample_ratio 0.25 \
#     --downsample_qrnn_adapter True \
#     --checkpoint_path /vision/u/eatang/sdas/sdqes/nnz7jx4b/checkpoints/ckpt-epoch-epoch=49.ckpt
