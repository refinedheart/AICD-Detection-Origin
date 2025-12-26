#!/bin/bash
/home/bai/anaconda3/envs/jittor-yolov5/bin/python train.py --data coco128.yaml --weights '' --cfg yolov5s.yaml --distill yolov5l.pt --img 416 --noplots --batch-size 8