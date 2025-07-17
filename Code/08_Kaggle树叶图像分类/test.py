import tensorflow as tf
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os


def get_file_paths(directory):
    """获取指定目录下所有文件的绝对路径"""
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

def decode(example):
    feature_description = {
        'target': tf.io.FixedLenFeature([], tf.int64),
        'image_name': tf.io.FixedLenFeature([], tf.string),
        'image': tf.io.FixedLenFeature([], tf.string),
    }
    feature_dict = tf.io.parse_single_example(example, feature_description)
    feature_dict['image'] = tf.io.decode_jpeg(feature_dict['image'])

    return feature_dict


if __name__ == "__main__":
    directory = "./data/train_tfrecords"
    file_paths = get_file_paths(directory)
    raw_dataset = tf.data.TFRecordDataset(file_paths)
    pharse_dataset = raw_dataset.map(decode)
    
    for raw_record in pharse_dataset.take(3):
        print(repr(raw_record))
