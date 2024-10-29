CUDA_VISIBLE_DEVICES=2,3 python train_net.py --config-file configs/faster_rcnn_R_101_C4_3x_caltech.yaml \
       	--num-gpus 2 OUTPUT_DIR output/Caltech_Pedestrians
