# 本文件用于xml标签格式转换为YOLO格式
import copy
# from lxml.etree import Element, SubElement, tostring, ElementTree

import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

classes = ["harbor", "ship", "storagetank", "chimney", "dam", "trainstation", "basketballcourt", "airport",
           "Expressway-Service-area", "airplane", "baseballfield", "Expressway-toll-station", "vehicle",
           "golffield", "bridge", "groundtrackfield", "overpass", "windmill", "tenniscourt", "stadium"]  # 目标类别

# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(image_id):
    in_file = open('/home/chenjiajie/桌面/Remo_detection/datasets/DIOR/Annotations/%s.xml' % image_id, encoding='UTF-8')  # xml文件路径
    out_file = open('/home/chenjiajie/桌面/Remo_detection/datasets/DIOR/labels/%s.txt' % image_id, 'w')  # 生成txt格式文件
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')

    # print('hello')
    w = int(size.find('width').text)
    # print('hello', w)
    h = int(size.find('height').text)
    # print(h)

    # w = 3000
    # h = 3000

    for obj in root.iter('object'):
        cls = obj.find('name').text
        # print(cls)
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


# xml_path = os.path.join(CURRENT_DIR, '/home/cjj/Desktop/Yolov5/Remodata/Annotations')

# xml list
img_xmls = os.listdir('/home/chenjiajie/桌面/Remo_detection/datasets/DIOR/Annotations')
for img_xml in img_xmls:
    label_name = img_xml.split('.')[0]
    print(label_name)
    convert_annotation(label_name)
