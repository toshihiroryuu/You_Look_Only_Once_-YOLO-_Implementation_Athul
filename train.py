import argparse
import os
import json
from preprocessing import parse_annotation
import numpy as np
from frontend import YOLO
from matplotlib import pyplot as plt

train_image_folder="RBC_datasets/JPEGImages/"
train_annot_folder="RBC_datasets/Annotations/"

saved_weights_name="models/tiny_yolo.h5"
labels=["RBC"]

nb_epoch=100
learning_rate=1e-4
batch_size=1
warmup_batches=250

object_scale=5.0
no_object_scale=1.0
coord_scale=1.0
class_scale=1.0

train_times=5
valid_times=1

architecture="Tiny Yolo"
input_size=416
max_box_per_img=10
anchors=[0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]


imgs,labels=parse_annotation(train_annot_folder,train_image_folder,labels)

train_valid_split=int(0.8*len(imgs))
np.random.shuffle(imgs)

valid_imgs=imgs[train_valid_split:]
train_imgs=imgs[:train_valid_split]

overlap_labels=set(labels).intersection(set(labels.keys()))

# print("Seen labels: "+str(labels))
# print("Given labels: "+str(labels)
print("Overelap labels: "+str(overlap_labels))

if len(overlap_labels)<len(labels):
  print("Some labels have no image! Please check it.")
  
	
yolo=YOLO(architecture=architecture,
          input_size=input_size,
          labels=labels,
          max_box_per_img=max_box_per_img,
          anchors=anchors)

yolo.train(
    train_imgs,
    valid_imgs,
    train_times,
    valid_times,
    nb_epoch,
    learning_rate,
    batch_size,
    warmup_batches,
    object_scale,
    no_object_scale,
    coord_scale,
    class_scale,
    saved_weights_name=saved_weights_name)
  

