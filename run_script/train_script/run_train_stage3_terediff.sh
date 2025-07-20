
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --main_process_port 10009 train.py       --config configs/train/train_stage3_terediff.yaml \
                                                        --config_testr testr/configs/TESTR/TESTR_R_50_Polygon.yaml \
