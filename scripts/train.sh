CUDA_VISIBLE_DEVICES=2,3
NUM_GPUS=2

case $OPTION in
    caltech) 
        OUTPUT_DIR=output/Caltech_Pedestrians
        CFG=configs/faster_rcnn_R_101_C4_3x_caltech.yaml
    ;;
    tju)
        OUTPUT_DIR=output/TJU-Pedestrian-Traffic
        CFG=configs/faster_rcnn_R_101_C4_3x_tju.yaml
    ;;
    ?) echo "Incorrect usage (caltech/tju)."
    ;;
esac

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
python train_net.py \
	--config-file $CFG \
    --num-gpus $NUM_GPUS \
	OUTPUT_DIR $OUTPUT_DIR