# -*- coding: utf-8 -*-
"""
@Auth ：挂科边缘
@File ：xml转txt.py
@IDE ：PyCharm
@Motto :学习新思想，争做新青年
@Email ：179958974@qq.com
"""
import xml.etree.ElementTree as ET
import os, cv2
import numpy as np


classes = []


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(xmlpath, xmlname):
    with open(xmlpath, "r", encoding='utf-8') as in_file:
        txtname = xmlname[:-4] + '.txt'
        txtfile = os.path.join(txtpath, txtname)
        tree = ET.parse(in_file)
        root = tree.getroot()
        filename = root.find('filename')
        img = cv2.imdecode(np.fromfile('{}/{}.{}'.format(imgpath, xmlname[:-4], postfix), np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        res = []
        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in classes:
                classes.append(cls)
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            bb = convert((w, h), b)
            res.append(str(cls_id) + " " + " ".join([str(a) for a in bb]))
        if len(res) != 0:
            with open(txtfile, 'w+') as f:
                f.write('\n'.join(res))


if __name__ == "__main__":
    postfix = 'png'  # 图像后缀
    imgpath = r'E:\A-毕业设计代做数据\helmet\test\images'  # 图像文件路径
    xmlpath = r'E:\A-毕业设计代做数据\helmet\test\annotations'  # xml文件文件路径
    txtpath = r'E:\A-毕业设计代做数据\helmet\test\labels'  # 生成的txt文件路径

    if not os.path.exists(txtpath):
        os.makedirs(txtpath, exist_ok=True)

    list = os.listdir(xmlpath)
    error_file_list = []
    for i in range(0, len(list)):
        try:
            path = os.path.join(xmlpath, list[i])
            if ('.xml' in path) or ('.XML' in path):
                convert_annotation(path, list[i])
                print(f'file {list[i]} convert success.')
            else:
                print(f'file {list[i]} is not xml format.')
        except Exception as e:
            print(f'file {list[i]} convert error.')
            print(f'error message:\n{e}')
            error_file_list.append(list[i])
    print(f'this file convert failure\n{error_file_list}')
    print(f'Dataset Classes:{classes}')
