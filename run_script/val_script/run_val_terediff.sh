
CUDA_VISIBLE_DEVICES=0 accelerate launch val.py         --config configs/val/val_terediff.yaml \
                                                        --config_testr testr/configs/TESTR/TESTR_R_50_Polygon.yaml \
