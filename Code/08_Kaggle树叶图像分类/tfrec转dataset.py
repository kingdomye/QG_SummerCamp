import os
import torch
import numpy as np
from torch import nn
import pandas as pd
import tensorflow as tf
from torchvision import models


def get_file_paths(directory="./data/train_tfrecords"):
    """获取指定目录下所有文件的绝对路径"""
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

def decode(example):
    """解析TFRecord中的数据"""
    feature_description = {
        'target': tf.io.FixedLenFeature([], tf.int64),
        'image_name': tf.io.FixedLenFeature([], tf.string),
        'image': tf.io.FixedLenFeature([], tf.string),
    }
    feature_dict = tf.io.parse_single_example(example, feature_description)
    feature_dict['image'] = tf.io.decode_jpeg(feature_dict['image'])

    return feature_dict

def train(dataset, model, loss_fn, optimizer, epochs):
    """训练模型"""
    model.train()
    for _ in range(epochs):
        for batch in dataset:
            images = batch['image']
            labels = batch['target']

            images_np_array = images.numpy()
            images_torch_tensor = torch.from_numpy(images_np_array)
            images_torch_tensor = images_torch_tensor.permute(0, 3, 1, 2)
            images_torch_tensor = images_torch_tensor.float()

            labels_np_array = labels.numpy()
            labels_torch_tensor = torch.from_numpy(labels_np_array)

            optimizer.zero_grad()
            outputs = model(images_torch_tensor)
            loss = loss_fn(outputs, labels_torch_tensor)
            loss.backward()
            optimizer.step()

            print('batch loss: ', loss.item())

def test(dataset, model):
    test_results = []
    """测试模型"""
    model.eval()
    with torch.no_grad():
        for batch in dataset:
            print('test')
            # images = batch['image']
            # image_name = batch['image_name']

            # images_np_array = images.numpy()
            # images_torch_tensor = torch.from_numpy(images_np_array)
            # images_torch_tensor = images_torch_tensor.permute(0, 3, 1, 2)
            # images_torch_tensor = images_torch_tensor.float()

            # image_name = image_name.numpy()
            # image_name = image_name[0]
            # image_name = image_name.decode('utf-8')

            # outputs = model(images_torch_tensor)
            # _, predicted = torch.max(outputs, 1)

            # predicted = predicted[0].item()

            # res = [image_name, predicted]
            # test_results.append(res)

    return test_results


# 模型定义
model = models.efficientnet_b4(weights=None)
num_classes = 5
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)


if __name__ == "__main__":
    epochs = 1
    batch_size = 1
    learning_rate = 1e-3

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    tfrecord_files = get_file_paths("./data/train_tfrecords")
    dataset = tf.data.TFRecordDataset(tfrecord_files)
    dataset = dataset.map(decode).batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)   # 预取数据以加快训练速度

    # train(dataset, model, loss_fn, optimizer, epochs)

    test_file_path = get_file_paths("./data/test_tfrecords")
    test_dataset = tf.data.TFRecordDataset(test_file_path)
    test_dataset = test_dataset.map(decode).batch(batch_size)

    print(test_dataset)

    # test_results = test(test_dataset, model)
    # 转为df
    # df = pd.DataFrame(test_results, columns=['image_id', 'label'], index=None)
    # df.to_csv('./sample_submission.csv', index=False)
