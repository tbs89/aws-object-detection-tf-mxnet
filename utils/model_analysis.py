#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Data Manipulation and Analysis
import pandas as pd
import numpy as np

# File and Directory Management
import zipfile
import glob
import shutil
import os
import os.path
from os import path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import glob


# Others
import random


# In[2]:

def transform_dataset_tf(df, name):
    
    df['normalized_boxes'].fillna("[]", inplace=True)
    df['confidences'].fillna(0, inplace=True)
    
    df[f'image_id_{name}'] = df['image_path'].str.extract(r'([^/]+)\.jpg$')
    
    def extract_boxes(x):
        box = eval(x)
        if box and isinstance(box, list) and len(box) == 4:
            return box
        else:
            return [None, None, None, None]
    
    df['boxes_list'] = df['normalized_boxes'].apply(extract_boxes)
    df[f'xmin_{name}'], df[f'ymin_{name}'], df[f'xmax_{name}'], df[f'ymax_{name}'] = zip(*df['boxes_list'])
    
    df[f'confidences_{name}'] = df['confidences']
    
    df_new = df[[f'image_id_{name}', f'xmin_{name}', f'ymin_{name}', f'xmax_{name}', f'ymax_{name}', f'confidences_{name}']].copy()
    
    return df_new




def transform_dataset_mx(df, name):
    
    df[f'image_id_{name}'] = df['image_path'].str.extract(r'([^/]+)\.jpg$')
    df[f'xmin_{name}'], df[f'ymin_{name}'], df[f'xmax_{name}'], df[f'ymax_{name}'] = zip(*df['normalized_boxes'].apply(lambda x: eval(x)[0]))
    df[f'confidences_{name}'] = df['confidences'].apply(lambda x: eval(x)[0])
    df_new = df[[f'image_id_{name}', f'xmin_{name}', f'ymin_{name}', f'xmax_{name}', f'ymax_{name}', f'confidences_{name}']].copy()
    
    return df_new


def calculate_iou(box1, box2):
    
    xi1 = max(box1['x1'], box2['x1'])
    yi1 = max(box1['y1'], box2['y1'])
    xi2 = min(box1['x2'], box2['x2'])
    yi2 = min(box1['y2'], box2['y2'])
    
    inter_area = max(0, xi2 - xi1 + 1) * max(0, yi2 - yi1 + 1)
    box1_area = (box1['x2'] - box1['x1'] + 1) * (box1['y2'] - box1['y1'] + 1)
    box2_area = (box2['x2'] - box2['x1'] + 1) * (box2['y2'] - box2['y1'] + 1)
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / union_area
    return iou


def calculate_iou_for_row(row):
    gt_box = {
        "x1": row["XMin"],
        "y1": row["YMin"],
        "x2": row["XMax"],
        "y2": row["YMax"]
    }
    tf_box = {
        "x1": row["xmin_tf"],
        "y1": row["ymin_tf"],
        "x2": row["xmax_tf"],
        "y2": row["ymax_tf"]
    }
    mx_box = {
        "x1": row["xmin_mx"],
        "y1": row["ymin_mx"],
        "x2": row["xmax_mx"],
        "y2": row["ymax_mx"]
    }
    iou_tf = calculate_iou(gt_box, tf_box)
    iou_mx = calculate_iou(gt_box, mx_box)
    
    return iou_tf, iou_mx


def draw_bounding_boxes(image_path, row):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Draw ground truth bounding box
    gt_box = [
        int(row["XMin"] * image.shape[1]),
        int(row["YMin"] * image.shape[0]),
        int(row["XMax"] * image.shape[1]),
        int(row["YMax"] * image.shape[0])
    ]
    cv2.rectangle(image, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), (0, 255, 0), 2)
    cv2.putText(image, 'Ground Truth', (gt_box[0], gt_box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Tensorflow bounding boxes
    if not np.isnan(row["xmin_tf"]):
        tf_box = [
            int(row["xmin_tf"] * image.shape[1]),
            int(row["ymin_tf"] * image.shape[0]),
            int(row["xmax_tf"] * image.shape[1]),
            int(row["ymax_tf"] * image.shape[0])
        ]
        cv2.rectangle(image, (tf_box[0], tf_box[1]), (tf_box[2], tf_box[3]), (255, 0, 0), 2)
        confidence_tf = round(row["confidences_tf"], 2)
        cv2.putText(image, f'TensorFlow: {confidence_tf}', (tf_box[0], tf_box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
    # MXNet bounding boxes
    if not np.isnan(row["xmin_mx"]):
        mx_box = [
            int(row["xmin_mx"] * image.shape[1]),
            int(row["ymin_mx"] * image.shape[0]),
            int(row["xmax_mx"] * image.shape[1]),
            int(row["ymax_mx"] * image.shape[0])
        ]
        cv2.rectangle(image, (mx_box[0], mx_box[1]), (mx_box[2], mx_box[3]), (0, 0, 255), 2)
        confidence_mx = round(row["confidences_mx"], 2)
        cv2.putText(image, f'MXNet: {confidence_mx}', (mx_box[0], mx_box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    plt.imshow(image)
    plt.axis('off')
    plt.show()


    
    
    
    
def calculate_mAP(df, confidence_col, iou_col):
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    all_aps = []

    for iou_thresh in iou_thresholds:

        sorted_data = df.sort_values(by=confidence_col, ascending=False)
        
        true_positives = []
        false_positives = []
        num_gt_boxes = len(df)
        
        for index, row in sorted_data.iterrows():
            if row[iou_col] > iou_thresh:
                true_positives.append(1)
                false_positives.append(0)
            else:
                true_positives.append(0)
                false_positives.append(1)

        true_positives = np.array(true_positives)
        false_positives = np.array(false_positives)

        # Precision and recall
        tp_cumsum = np.cumsum(true_positives)
        fp_cumsum = np.cumsum(false_positives)
        
        recalls = tp_cumsum / float(num_gt_boxes)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

        
        recalls = np.concatenate(([0.], recalls, [1.]))
        precisions = np.concatenate(([0.], precisions, [0.]))

        
        for i in range(precisions.size - 1, 0, -1):
            precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])

        
        ap = np.sum((recalls[1:] - recalls[:-1]) * precisions[:-1])
        all_aps.append(ap)

    return np.mean(all_aps)




def precision_recall_curve(df, confidence_col, iou_col, iou_threshold):
    sorted_data = df.sort_values(by=confidence_col, ascending=False)

    true_positives = []
    false_positives = []
    num_gt_boxes = len(df)

    for index, row in sorted_data.iterrows():
        if row[iou_col] > iou_threshold:
            true_positives.append(1)
            false_positives.append(0)
        else:
            true_positives.append(0)
            false_positives.append(1)

    true_positives = np.array(true_positives)
    false_positives = np.array(false_positives)

    
    tp_cumsum = np.cumsum(true_positives)
    fp_cumsum = np.cumsum(false_positives)

    recalls = tp_cumsum / float(num_gt_boxes)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

    recalls = np.concatenate(([0.], recalls, [1.]))
    precisions = np.concatenate(([0.], precisions, [0.]))

    for i in range(precisions.size - 1, 0, -1):
        precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])

    return recalls, precisions