# 全部流程代码
import re
from collections import Counter

import numpy as np
import torch
from torchgen.api.types import longT
from transformers import BertTokenizer
import os
from transformers import RobertaTokenizer, RobertaModel

tokenizer = RobertaTokenizer.from_pretrained('../model/codebert-base')

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):
        self.labels = []
        self.texts = []
        for index, row in df.iterrows():
            label = row['Class'] - 1
            text = row['miniopcode']
            opcodes = text.split()
            counts = Counter(opcodes)
            tokenized_text = tokenizer(text,
                                        padding='max_length',
                                        max_length=512,
                                        truncation=True,
                                        return_tensors="pt")
            # 生成occurrence_mask
            input_ids = tokenized_text['input_ids'].squeeze(0)
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            occurrence_mask = []
            for token in tokens:
                # cleaned_token = token.replace('Ġ', '')  # 处理RoBERTa前缀
                cleaned_token = re.sub(r'^Ġ', '', token)  # 使用正则移除前缀
                occurrence = counts.get(cleaned_token, 0)
                occurrence_mask.append(1 if occurrence == 1 else 0)
            tokenized_text['occurrence_mask'] = torch.tensor(occurrence_mask, dtype=torch.float32).unsqueeze(0)
            self.texts.append(tokenized_text)
            self.labels.append(int(label))
            if index < 3:  # 检查前3个样本
                print("\n=== 原始操作码序列 ===")
                print(opcodes[:10])  # 打印前10个操作码
                print("=== 分词结果 ===")
                print(tokens[:10])  # 打印前10个token
                print("=== occurrence_mask ===")
                print(occurrence_mask[:10])  # 打印前10个mask值

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y


import pandas as pd

subtrain_text_df = pd.read_csv('../data/strain_miniopcode.csv')
df = pd.DataFrame(subtrain_text_df)
np.random.seed(112)
df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42), [int(.8 * len(df)), int(.9 * len(df))])
print(len(df_train), len(df_val), len(df_test))

from torch import nn

class BertClassifier(nn.Module):
    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.bert = RobertaModel.from_pretrained('../model/codebert-base')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 9)
        self.relu = nn.ReLU()

        # 关键修正：使用 nn.Parameter 注册可学习参数
        self.alpha = nn.Parameter(torch.tensor(5.0))  # 正确注册方式

    def forward(self, input_id, mask,occurrence_mask):
        # 获取BERT的序列输出
        sequence_output, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)

        # 监控 alpha 值和权重分布
        print(f"Alpha: {self.alpha.item():.3f}")
        print("Occurrence_mask mean:", occurrence_mask.float().mean().item())



        # 检查加权效果
        weight_factor = (1 + self.alpha * occurrence_mask.unsqueeze(-1))
        print("Weight factor stats:")
        print("Min:", weight_factor.min().item())
        print("Max:", weight_factor.max().item())
        print("Mean:", weight_factor.mean().item())
        weighted_sequence = sequence_output * weight_factor
        # 调整出现一次的token的权重
        # weighted_sequence = sequence_output * (1 + self.alpha * occurrence_mask.unsqueeze(-1))

        # 使用CLS token作为分类特征
        cls_token = weighted_sequence[:, 0, :]
        dropout_output = self.dropout(cls_token)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer
        # _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        # dropout_output = self.dropout(pooled_output)
        # linear_output = self.linear(dropout_output)
        # final_layer = self.relu(linear_output)
        # return final_layer

from torch.optim import Adam
from tqdm import tqdm

def train(model, train_data, val_data, learning_rate, epochs):
    # 通过Dataset类获取训练和验证集
    train, val = Dataset(train_data), Dataset(val_data)
    # DataLoader根据batch_size获取数据，训练时选择打乱样本
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)
    # 判断是否使用GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    # 开始进入训练循环
    for epoch_num in range(epochs):
        # 定义两个变量，用于存储训练集的准确率和损失
        total_acc_train = 0
        total_loss_train = 0
        # 进度条函数tqdm
        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            # mask = train_input['attention_mask'].to(device)
            # input_id = train_input['input_ids'].squeeze(1).to(device)
            mask = train_input['attention_mask'].squeeze(1).to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)
            occurrence_mask = train_input['occurrence_mask'].squeeze(1).to(device)
            # 通过模型得到输出
            # output = model(input_id, mask)
            output = model(input_id, mask, occurrence_mask)
            # 计算损失
            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item()
            # 计算精度
            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc
            # 模型更新
            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
        # ------ 验证模型 -----------
        # 定义两个变量，用于存储验证集的准确率和损失
        total_acc_val = 0
        total_loss_val = 0
        # 不需要计算梯度
        with torch.no_grad():
            # 循环获取数据集，并用训练好的模型进行验证
            for val_input, val_label in val_dataloader:
                # 如果有GPU，则使用GPU，接下来的操作同训练
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)
                # 新增：提取 occurrence_mask
                occurrence_mask = val_input['occurrence_mask'].squeeze(1).to(device)

                output = model(input_id, mask, occurrence_mask)

                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        print(
            f'''Epochs: {epoch_num + 1} 
              | Train Loss: {total_loss_train / len(train_data): .3f} 
              | Train Accuracy: {total_acc_train / len(train_data): .3f} 
              | Val Loss: {total_loss_val / len(val_data): .3f} 
              | Val Accuracy: {total_acc_val / len(val_data): .3f}''')

# 因为本案例中是处理多类分类问题，则使用分类交叉熵作为我们的损失函数。
EPOCHS = 30
model = BertClassifier()
LR = 2e-5
train(model, df_train, df_val, LR, EPOCHS)



# 测试模型
def evaluate(model, test_data):
    test = Dataset(test_data)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        model = model.cuda()

    total_acc_test = 0
    with torch.no_grad():
        for test_input, test_label in test_dataloader:
            test_label = test_label.to(device)
            # mask = test_input['attention_mask'].to(device)
            # input_id = test_input['input_ids'].squeeze(1).to(device)
            # output = model(input_id, mask)
            mask = test_input['attention_mask'].squeeze(1).to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)
            occurrence_mask = test_input['occurrence_mask'].squeeze(1).to(device)
            output = model(input_id, mask, occurrence_mask)
            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc
    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')

# 用测试数据集进行测试
evaluate(model, df_test)
torch.save(model.state_dict(),"berttextlong1.pth")