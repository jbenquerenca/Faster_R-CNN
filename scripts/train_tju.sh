CUDA_VISIBLE_DEVICES=2,3 python train_net.py --config-file configs/faster_rcnn_R_101_C4_3x.yaml \
       	--num-gpus 2 --resume OUTPUT_DIR output/TJU-Pedestrian-Traffic
