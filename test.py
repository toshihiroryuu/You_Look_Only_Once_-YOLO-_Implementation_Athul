from frontend import YOLO
import cv2
import argparse
import os
import json
from utils import draw_boxes
from matplotlib import pyplot as plt
import numpy as np
import time



valid_times=1
train_annot_folder="RBC_datasets/Annotations/"
train_image_folder="RBC_datasets/JPEGImages/"
train_times=1
batch_size=16
learning_rate=1e-4
nb_epoch=50
warmup_batches=250
object_scale=5.0
no_object_scale=1.0
coord_scale=1.0
class_scale=1.0
labels=["RBC"]
input_size=416
max_box_per_img=10
anchors=[0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
architecture="Tiny Yolo"
grid_w=13
grid_h=13

weights_path="models/tiny_yolo.h5"
test_path="/home/quest/yolo2/RBC_datasets/test"
out_dir="/home/quest/yolo2/output/"
  
  
yolo=YOLO(architecture=architecture,
          input_size=input_size,
          labels=labels,
          max_box_per_img=max_box_per_img,
          anchors=anchors)
yolo.load_weights(weights_path)

for f in os.listdir(test_path):
  print(f)
  f_path=os.path.join(test_path,f)

  img=cv2.imread(f_path)
  boxes=yolo.predict(img)
  
  img=draw_boxes(boxes,img,labels)
  cv2.imwrite(out_dir+f_path[-9:],img)
     
