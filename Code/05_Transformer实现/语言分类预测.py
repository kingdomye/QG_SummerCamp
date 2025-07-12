# ================================
# @File         : 语言分类预测.py
# @Time         : 2025/07/12
# @Author       : Yingrui Chen
# @description  : 利用transformer实现语言分类预测
#                 数据集采用酒店评论，可以对酒店进行情感分类
# ================================

import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 设备配置
tokenizer = AutoTokenizer.from_pretrained("./data/rbt3")
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class Dataset(Dataset):
    def __init__(self):
        super().__init__()

        self.data = pd.read_csv('./data/ChnSentiCorp_htl_all.csv')
        self.data = self.data.dropna()

    def __getitem__(self, index):
        review = self.data.iloc[index]["review"]
        label = self.data.iloc[index]["label"]
        return review, label
    
    def __len__(self):
        return len(self.data)
    

def collate_func(batch):
    texts, labels = [], []
    for item in batch:
        texts.append(item[0])
        labels.append(item[1])

    inputs = tokenizer(texts, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
    inputs['labels'] = torch.tensor(labels)

    return inputs


def train(epoch=3, log_step=100, optimizer=None):
    global_step = 0
    for ep in range(epoch):
        model.train()
        for batch in trainloader:
            # 将数据移动到指定设备
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            output = model(**batch)
            output.loss.backward()
            optimizer.step()
            if global_step % log_step == 0:
                print(f"epoch: {ep}, global_step: {global_step}, loss: {output.loss.item()}")
            global_step += 1
        acc = evaluate()
        print(f"ep: {ep}, acc: {acc}")


def evaluate():
    model.eval()
    acc_num = 0
    with torch.inference_mode():
        for batch in validloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(**batch)
            pred = torch.argmax(output.logits, dim=-1)
            acc_num += (pred.long() == batch["labels"].long()).float().sum()
    return acc_num / len(valid_set)


if __name__ == '__main__':
    ds = Dataset()
    train_set, valid_set = random_split(ds, lengths=[0.9, 0.1])
    trainloader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=collate_func)
    validloader = DataLoader(valid_set, batch_size=64, shuffle=False, collate_fn=collate_func)

    # 将模型移动到指定设备
    model = AutoModelForSequenceClassification.from_pretrained("./data/rbt3").to(device)

    sen = input("请输入评论：")
    id2_label = {0: "差评！", 1: "好评！"}

    model.eval()
    with torch.inference_mode():
        inputs = tokenizer(sen, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=-1)
        print(f"模型预测结果:{id2_label.get(pred.item())}")
