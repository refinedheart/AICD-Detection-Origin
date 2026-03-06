#!/bin/bash
python train.py --data BCCD.yaml --cfg yolov5l.yaml --weights yolov5l.pt --batch-size 8 --epoch 100 --name teacher_bccd
# /home/bai/anaconda3/envs/jittor-yolov5/bin/python train.py --data coco128.yaml --weights '' --cfg yolov5s.yaml --distill yolov5l.pt --img 416 --noplots --batch-size 8

# python train.py --data BCCD.yaml --cfg yolov5s.yaml --weights '' --batch-size 16 --epoch 300 --distill runs/train/teacher_bccd/weights/best.pt --name student_fgfi
