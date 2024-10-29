#!/bin/bash
outputs_folder="output/Caltech_Pedestrians"
ckpts_folder=$outputs_folder/eval_ckpts
config_file="configs/faster_rcnn_R_101_C4_3x_caltech.yaml"
ckpts=()
for file in "$outputs_folder"/*; do
    if [ -f "$file" ]; then
        if [[ "$file" == *".pth"* ]]; then
            ckpts+=("$file")
        fi
    fi
done
if [ ${#ckpts[@]} -eq 0 ]; then
    echo "No models found."
    exit 0
fi
for model in "${ckpts[@]}"; do
    filename=$(basename $model)
    echo "Evaluating $filename"
    python train_net.py --config-file ${config_file} --eval-only OUTPUT_DIR $ckpts_folder/$filename MODEL.WEIGHTS $model
done