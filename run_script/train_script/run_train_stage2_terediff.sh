
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch train.py       --config configs/train/train_stage2_terediff.yaml \
                                                        --config_testr testr/configs/TESTR/TESTR_R_50_Polygon.yaml \
