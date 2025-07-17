# ================================
# @File         : 文件Feature获取.py
# @Time         : 2025/07/17
# @Author       : Yingrui Chen
# @description  : 获取TFRecord文件的特征结构和样本内容
# ================================

import tensorflow as tf

def inspect_tfrecord(tfrecord_file):
    """查看TFRecord文件的结构和样本内容"""
    # 创建TFRecord数据集
    raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
    
    # 获取第一个样本
    for raw_record in raw_dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        
        # 打印特征名称和类型
        print("===== TFRecord 特征结构 =====")
        for feature_name, feature in example.features.feature.items():
            if feature.HasField('bytes_list'):
                dtype = 'bytes_list'
                value = feature.bytes_list.value
            elif feature.HasField('float_list'):
                dtype = 'float_list'
                value = feature.float_list.value
            elif feature.HasField('int64_list'):
                dtype = 'int64_list'
                value = feature.int64_list.value
            else:
                dtype = 'unknown'
                value = []
                
            # 只显示前5个值，避免输出过长
            short_value = list(value)[:5]
            if len(value) > 5:
                short_value.append('...')
                
            print(f"特征名: {feature_name}")
            print(f"  类型: {dtype}")
            print(f"  示例值: {short_value}")
            print()
        
        print("===== 提示 =====")
        print("根据以上输出，你可以为TFRecordDataset创建如下特征描述:")
        print("feature_description = {")
        for feature_name, feature in example.features.feature.items():
            if feature.HasField('bytes_list'):
                dtype = 'tf.string'
            elif feature.HasField('float_list'):
                dtype = 'tf.float32'
            elif feature.HasField('int64_list'):
                dtype = 'tf.int64'
            else:
                dtype = 'unknown_type'
                
            print(f"    '{feature_name}': tf.io.FixedLenFeature([], {dtype}),")
        print("}")

# 使用示例
if __name__ == "__main__":
    tfrecord_file = "./data/test_tfrecords/ld_test00-1.tfrec"  # 替换为你的TFRecord文件路径
    inspect_tfrecord(tfrecord_file)