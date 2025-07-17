import os
import torch
import numpy as np
import pandas as pd
import tensorflow as tf
from torch import nn
from torchvision import models
from torch.optim.lr_scheduler import StepLR


# --------------------------
# 1. 配置与工具函数
# --------------------------
# 限制TensorFlow使用CPU（避免与PyTorch的CUDA冲突）
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# 设备配置（自动使用GPU if available）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# EfficientNet输入归一化参数（匹配ImageNet分布）
MEAN = torch.tensor([0.485, 0.456, 0.406]).to(device)  # RGB均值
STD = torch.tensor([0.229, 0.224, 0.225]).to(device)   # RGB标准差


def get_file_paths(directory="./data/train_tfrecords"):
    """获取目录下所有文件的绝对路径"""
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".tfrecord"):  # 只保留tfrecord文件
                file_paths.append(os.path.join(root, file))
    return file_paths


# --------------------------
# 2. 数据处理（解析+增强+转换）
# --------------------------
def decode(example):
    """解析TFRecord，返回(image, target, image_name)"""
    feature_description = {
        'target': tf.io.FixedLenFeature([], tf.int64),
        'image_name': tf.io.FixedLenFeature([], tf.string),
        'image': tf.io.FixedLenFeature([], tf.string),
    }
    feature_dict = tf.io.parse_single_example(example, feature_description)
    image = tf.io.decode_jpeg(feature_dict['image'])  # 解码为[H, W, 3] uint8
    image = tf.image.resize(image, (224, 224))  # 统一尺寸为EfficientNet输入
    return image, feature_dict['target'], feature_dict['image_name']


def train_augment(image):
    """训练数据增强"""
    # 随机水平翻转
    image = tf.image.random_flip_left_right(image)
    # 随机亮度调整
    image = tf.image.random_brightness(image, max_delta=0.2)
    # 随机对比度调整
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    return image


def tf2torch(image_tf, target_tf=None):
    """
    将TensorFlow张量转换为PyTorch张量（零拷贝优化）
    - 图像: [H, W, 3] uint8 -> [C, H, W] float32 (归一化后)
    - 标签: 整数张量
    """
    # 图像转换
    # 移到CPU（若在GPU）并转为DLPack
    if image_tf.device.endswith('GPU'):
        image_tf = image_tf.cpu()
    image_torch = torch.from_dlpack(tf.experimental.dlpack.to_dlpack(image_tf))
    # 维度转换: [H, W, C] -> [C, H, W]
    image_torch = image_torch.permute(2, 0, 1).float()
    # 归一化: [0,255] -> [0,1] 并减去均值/除以标准差
    image_torch = image_torch / 255.0
    image_torch = (image_torch - MEAN[:, None, None]) / STD[:, None, None]

    # 标签转换（若存在）
    if target_tf is not None:
        if target_tf.device.endswith('GPU'):
            target_tf = target_tf.cpu()
        target_torch = torch.from_dlpack(tf.experimental.dlpack.to_dlpack(target_tf))
        return image_torch.to(device), target_torch.to(device)
    else:
        return image_torch.to(device)


# --------------------------
# 3. 训练与测试函数优化
# --------------------------
def train(dataset, model, loss_fn, optimizer, scheduler, epochs=1, log_interval=10):
    """训练函数（添加学习率调度、进度日志）"""
    model.train()
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        batch_idx = 0
        total_loss = 0.0
        for batch in dataset:
            images_tf, targets_tf, _ = batch  # 从dataset获取增强后的数据
            
            # 转换为PyTorch张量（批量处理）
            images_torch = []
            targets_torch = []
            for img, tgt in zip(images_tf, targets_tf):
                img_t, tgt_t = tf2torch(img, tgt)
                images_torch.append(img_t)
                targets_torch.append(tgt_t)
            images_torch = torch.stack(images_torch)
            targets_torch = torch.stack(targets_torch)

            # 前向+反向传播
            optimizer.zero_grad()
            outputs = model(images_torch)
            loss = loss_fn(outputs, targets_torch)
            loss.backward()
            optimizer.step()

            # 日志记录
            total_loss += loss.item()
            batch_idx += 1
            if batch_idx % log_interval == 0:
                avg_loss = total_loss / log_interval
                print(f"Batch {batch_idx} | Avg Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
                total_loss = 0.0
        
        # 每个epoch结束后更新学习率
        scheduler.step()


def test(dataset, model, num_batches=5):
    """测试函数（支持任意batch_size，修复样本遍历逻辑）"""
    model.eval()
    test_results = []
    with torch.no_grad():
        batch_idx = 0
        for batch in dataset:
            if batch_idx >= num_batches:
                break
            images_tf, _, image_names_tf = batch  # 获取图像和文件名
            
            # 转换图像
            images_torch = [tf2torch(img) for img in images_tf]
            images_torch = torch.stack(images_torch)

            # 推理
            outputs = model(images_torch)
            _, predicted = torch.max(outputs, 1)  # 取预测类别

            # 处理文件名（批量解析）
            for img_name_tf, pred in zip(image_names_tf, predicted):
                img_name = img_name_tf.numpy().decode('utf-8')  # bytes -> str
                test_results.append([img_name, pred.item()])
            
            batch_idx += 1
    return test_results


# --------------------------
# 4. 主函数（流程优化）
# --------------------------
if __name__ == "__main__":
    # 配置参数
    epochs = 3
    batch_size = 16  # 增大batch_size提升GPU利用率
    learning_rate = 1e-4  # 较小的初始学习率
    log_interval = 20  # 每20个batch打印一次日志

    # 1. 加载并预处理数据
    tfrecord_files = get_file_paths("./data/train_tfrecords")
    # 训练集：解析->增强->批量处理
    train_dataset = (
        tf.data.TFRecordDataset(tfrecord_files)
        .map(decode, num_parallel_calls=tf.data.AUTOTUNE)  # 并行解析
        .map(lambda img, tgt, name: (train_augment(img), tgt, name), num_parallel_calls=tf.data.AUTOTUNE)  # 并行增强
        .shuffle(1024)  # 打乱数据
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)  # 预取数据
    )
    # 测试集（这里复用训练集演示，实际应使用独立测试集）
    test_dataset = (
        tf.data.TFRecordDataset(tfrecord_files)
        .map(decode, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    # 2. 初始化模型、损失函数、优化器
    model = models.efficientnet_b4(weights=None)
    num_classes = 5
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model = model.to(device)  # 移到GPU/CPU

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # 添加权重衰减
    scheduler = StepLR(optimizer, step_size=1, gamma=0.9)  # 学习率衰减

    # 3. 训练与测试
    train(train_dataset, model, loss_fn, optimizer, scheduler, epochs, log_interval)
    test_results = test(test_dataset, model, num_batches=5)

    # 4. 保存结果
    df = pd.DataFrame(test_results, columns=['image_id', 'label'])
    df.to_csv('./sample_submission.csv', index=False)
    print("测试结果已保存到sample_submission.csv")

    # 5. 保存模型
    torch.save(model.state_dict(), 'efficientnet_b4_model.pth')
    print("模型已保存到efficientnet_b4_model.pth")