# -*- coding: utf-8 -*-
"""
@Auth ： 挂科边缘
@File ：detect.py
@IDE ：PyCharm
@Motto:学习新思想，争做新青年
@Email ：179958974@qq.com
"""

from ultralytics import YOLO
import cv2

if __name__ == '__main__':

    # Load a model
    model = YOLO(model=r'D:\硕士准备\YOLOV13\yolov13-main\best.pt')
    model.predict(source=r'103236.mp4',
                  save=True,
                  show=True,
                  )