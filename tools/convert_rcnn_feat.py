"""
Modified from https://github.com/hengyuan-hu/bottom-up-attention-vqa/blob/master/tools/detection_features_converter.py
Reads in a tsv file with pre-trained bottom up attention features and
stores it in HDF5 format.  Also store {image_id: feature_idx}
 as a pickle file.
Hierarchy of HDF5 file:
{ 'image_features': num_images x num_boxes x 2048 array of features
  'image_bb': num_images x num_boxes x 4 array of bounding boxes }
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util.io import save_pickle
from tqdm import tqdm
import ipdb
import numpy as np
import csv
import base64
import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


csv.field_size_limit(sys.maxsize)

FIELDNAMES = ["image_id", "image_w", "image_h", "num_boxes", "boxes", "features"]
infile = "../data_bin/trainval_36/trainval_resnet101_faster_rcnn_genome_36.tsv"

feature_length = 2048
num_fixed_boxes = 36

if __name__ == "__main__":
    train_dir = "../data_bin/train2014/"
    val_dir = "../data_bin/val2014"

    def get_image_id(image_dir):
        filenames = os.listdir(image_dir)
        image_id = set(int(fn[-16:-4]) for fn in filenames)
        return image_id

    train_imgids = get_image_id(train_dir)
    val_imgids = get_image_id(val_dir)

    train_indices = {}
    val_indices = {}

    train_img_features = np.zeros((len(train_imgids), num_fixed_boxes, feature_length))
    train_img_bb = np.zeros((len(train_imgids), num_fixed_boxes, 4))
    train_spatial_img_features = np.zeros((len(train_imgids), num_fixed_boxes, 6))

    val_img_features = np.zeros((len(val_imgids), num_fixed_boxes, feature_length))
    val_img_bb = np.zeros((len(val_imgids), num_fixed_boxes, 4))
    val_spatial_img_features = np.zeros((len(val_imgids), num_fixed_boxes, 6))

    train_counter = 0
    val_counter = 0

    print("reading tsv...")
    with open(infile, "r") as tsv_in_file:
        # ipdb.set_trace()
        reader = csv.DictReader(tsv_in_file, delimiter="\t", fieldnames=FIELDNAMES)
        for item in tqdm(reader, miniters=1000):
            item["num_boxes"] = int(item["num_boxes"])
            image_id = int(item["image_id"])
            image_w = float(item["image_w"])
            image_h = float(item["image_h"])
            bboxes = np.frombuffer(
                base64.b64decode(item["boxes"]), dtype=np.float32
            ).reshape((item["num_boxes"], -1))

            box_width = bboxes[:, 2] - bboxes[:, 0]
            box_height = bboxes[:, 3] - bboxes[:, 1]
            scaled_width = box_width / image_w
            scaled_height = box_height / image_h
            scaled_x = bboxes[:, 0] / image_w
            scaled_y = bboxes[:, 1] / image_h

            box_width = box_width[..., np.newaxis]
            box_height = box_height[..., np.newaxis]
            scaled_width = scaled_width[..., np.newaxis]
            scaled_height = scaled_height[..., np.newaxis]
            scaled_x = scaled_x[..., np.newaxis]
            scaled_y = scaled_y[..., np.newaxis]

            spatial_features = np.concatenate(
                (
                    scaled_x,
                    scaled_y,
                    scaled_x + scaled_width,
                    scaled_y + scaled_height,
                    scaled_width,
                    scaled_height,
                ),
                axis=1,
            )

            img_features = np.frombuffer(
                base64.b64decode(item["features"]), dtype=np.float32
            ).reshape((item["num_boxes"], -1))

            cache_dict = {
                "img_features": img_features,
                "spatial_features": spatial_features,
                "bboxes": bboxes,
            }

            scope = "train" if image_id in train_imgids else "val"
            os.makedirs(f"../dataset/{scope}_img_feat", exist_ok=True)
            save_pickle(cache_dict, f"../dataset/{scope}_img_feat/{image_id}.pkl")

            # ipdb.set_trace()

    if len(train_imgids) != 0:
        print("Warning: train_image_ids is not empty")

    if len(val_imgids) != 0:
        print("Warning: val_image_ids is not empty")

    print("done!")
