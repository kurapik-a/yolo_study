# -*- coding: utf-8 -*-
"""
@Auth ： 挂科边缘
@File ：detect.py
@IDE ：PyCharm
@Motto:学习新思想，争做新青年
@Email ：179958974@qq.com
"""

from ultralytics import YOLO

if __name__ == '__main__':

    # Load a model
    model = YOLO(model=r'yolov12n.pt')
    model.predict(source=r'20241002024400.mp4',
                  save=True,
                  show=True,
                  )