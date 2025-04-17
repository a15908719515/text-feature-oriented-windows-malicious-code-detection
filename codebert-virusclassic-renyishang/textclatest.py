# 全部流程代码
import numpy as np
import torch
from torchgen.api.types import longT
from transformers import BertTokenizer
import os
from transformers import RobertaTokenizer, RobertaModel
import pandas as pd
from torch import nn
from collections import defaultdict
import math

tokenizer = RobertaTokenizer.from_pretrained('../model/codebert-base')
# ========== Rényi熵计算模块 ==========
def calculate_renyi_weights(df, alpha=2, top_k=5):
    """计算全局Rényi熵并返回top-k高熵操作码的权重字典"""
    class_op_counts = defaultdict(lambda: defaultdict(int))
    total_class_counts = defaultdict(int)

    for _, row in df.iterrows():
        if not isinstance(row["miniopcode"], str):
            continue
        ops = row["miniopcode"].split()
        cls = row["Class"] - 1  # 适配你的标签格式
        for op in set(ops):
            class_op_counts[cls][op] += 1
        total_class_counts[cls] += 1

    # 计算Rényi熵
    op_entropy = {}
    all_ops = set()
    for cls in class_op_counts.values():
        all_ops.update(cls.keys())

    for op in all_ops:
        probabilities = []
        for cls in total_class_counts.keys():
            count = class_op_counts[cls].get(op, 0)
            total = total_class_counts[cls]
            probabilities.append(count / total if total > 0 else 0)

        valid_probs = [p for p in probabilities if p > 0]
        if not valid_probs:
            op_entropy[op] = 0.0
            continue
        if alpha == 1:
            entropy = -sum(p * math.log(p, 2) for p in valid_probs)
        else:
            sum_p_alpha = sum(p ** alpha for p in valid_probs)
            entropy = (1 / (1 - alpha)) * math.log(sum_p_alpha, 2)
        op_entropy[op] = entropy

    # 选择top-k高熵操作码
    sorted_ops = sorted(op_entropy.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return {op: 3 for op, _ in sorted_ops}  # 权重增强系数设为3倍


# ========== 增强版BERT分类器 ==========
class RenyiBertClassifier(nn.Module):
    def __init__(self, renyi_weights, dropout=0.5):
        super().__init__()
        self.bert = RobertaModel.from_pretrained('../model/codebert-base', attn_implementation="eager")
        self.tokenizer = RobertaTokenizer.from_pretrained('../model/codebert-base')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 9)
        self.relu = nn.ReLU()


        # 初始化Rényi权重
        self.renyi_weights = renyi_weights
        self.enhance_factor = 1.5  # 注意力增强系数


        # 创建操作码到token的映射
        self.op_to_tokens = {
            op: self.tokenizer.encode(op, add_special_tokens=False)
            for op in renyi_weights.keys()
        }


    def create_enhance_mask(self, input_ids):
        """生成注意力增强掩码"""
        batch_size, seq_len = input_ids.shape
        mask = torch.ones((batch_size, seq_len), device=input_ids.device)

        # 转换token到操作码
        all_tokens = [self.tokenizer.convert_ids_to_tokens(ids) for ids in input_ids.cpu().numpy()]

        for b in range(batch_size):
            tokens = all_tokens[b]
            for pos, token in enumerate(tokens):
                # 检查是否是特殊token
                if token in self.tokenizer.special_tokens_map.values():
                    continue

                # 反向查找原始操作码
                for op, op_tokens in self.op_to_tokens.items():
                    if pos >= len(op_tokens):
                        continue
                    if tokens[pos:pos + len(op_tokens)] == op_tokens:
                        mask[b, pos:pos + len(op_tokens)] = self.enhance_factor
                        break
        return mask.unsqueeze(1)  # [batch, 1, seq_len]

    def forward(self, input_ids, attention_mask):
        # 生成增强掩码
        enhance_mask = self.create_enhance_mask(input_ids)

        # 扩展掩码维度 [batch, 1, seq_len] -> [batch, heads, seq_len, seq_len]
        enhance_mask = enhance_mask.unsqueeze(2).repeat(1, 1, attention_mask.size(-1), 1)

        # 修改注意力机制
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True,
            output_hidden_states = True,  # 必须添加这个参数
        )

        # # 增强最后一层的注意力
        # last_attention = outputs.attentions[-1]  # [batch, heads, seq, seq]
        # enhanced_attention = last_attention * enhance_mask.to(last_attention.device)
        last_attention = outputs.attentions[-1]  # [batch, heads, seq, seq]
        hidden_states = outputs.hidden_states[-1]  # [batch, seq, hidden_size]
        # 增强注意力并重新计算上下文
        enhanced_attention = last_attention * enhance_mask
        attention_probs = nn.functional.softmax(enhanced_attention, dim=-1)
        # 正确计算上下文表示
        context = torch.matmul(attention_probs, hidden_states.unsqueeze(1))  # [batch, heads, seq, hidden]
        # 池化策略：取所有注意力头的平均值
        pooled_output = context.mean(dim=2)  # [batch, heads, hidden]
        pooled_output = pooled_output.mean(dim=1)  # [batch, hidden]

        # 分类头
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        return self.relu(linear_output)  # 输出形状 [batch, num_classes]
        #
        # # 使用增强后的注意力重新计算上下文
        # # print(f"Outputs: {outputs}")  # 调试信息
        # hidden_states = outputs.hidden_states[-1]
        # attention_probs = nn.functional.softmax(enhanced_attention, dim=-1)
        # context = torch.matmul(attention_probs, hidden_states.unsqueeze(1))
        # pooled_output = context.mean(dim=2).squeeze(1)
        #
        # # 分类头
        # dropout_output = self.dropout(pooled_output)
        # linear_output = self.linear(dropout_output)
        # return self.relu(linear_output)

# ========== 修改后的训练流程 ==========
# 初始化Rényi权重
subtrain_text_df = pd.read_csv('../data/strain_miniopcode.csv')
renyi_weights = calculate_renyi_weights(subtrain_text_df, alpha=2)

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):
        self.labels = []
        self.texts = []
        for index, row in df.iterrows():
            label = row['Class'] - 1
            text = row['miniopcode']#miniopcode
            tokenized_text = tokenizer(text,
                                        padding='max_length',
                                        max_length=512,
                                        truncation=True,
                                        return_tensors="pt")
            self.texts.append(tokenized_text)
            self.labels.append(int(label))

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

subtrain_text_df = pd.read_csv('../data/strain_miniopcode.csv')#strain_miniopcode
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

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer




from torch.optim import Adam
from tqdm import tqdm


def train_enhanced(model, train_data, val_data, learning_rate, epochs):
    train, val = Dataset(train_data), Dataset(val_data)
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch in range(epochs):
        model.train()
        total_loss, total_acc = 0, 0

        for batch in tqdm(train_dataloader):
            inputs, labels = batch
            labels = labels.to(device)
            input_ids = inputs['input_ids'].squeeze(1).to(device)
            attention_mask = inputs['attention_mask'].squeeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)

            # 添加形状验证
            assert outputs.dim() == 2, f"输出维度异常: {outputs.shape}"
            assert outputs.size(1) == 9, f"类别数不符: {outputs.size(1)}"

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += (outputs.argmax(1) == labels).sum().item()

        # 验证循环
        model.eval()
        val_loss, val_acc = 0, 0
        with torch.no_grad():
            for batch in val_dataloader:
                inputs, labels = batch
                labels = labels.to(device)
                input_ids = inputs['input_ids'].squeeze(1).to(device)
                attention_mask = inputs['attention_mask'].squeeze(1).to(device)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_acc += (outputs.argmax(1) == labels).sum().item()

        print(f"Epoch {epoch + 1}")
        print(f"Train Loss: {total_loss / len(train):.4f} | Acc: {total_acc / len(train):.4f}")
        print(f"Val Loss: {val_loss / len(val):.4f} | Acc: {val_acc / len(val):.4f}")


# 初始化并训练模型
model = RenyiBertClassifier(renyi_weights)
EPOCHS = 20
LR = 1e-5
train_enhanced(model, df_train, df_val, LR, EPOCHS)



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
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)
            output = model(input_id, mask)
            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc
    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')

# 用测试数据集进行测试
evaluate(model, df_test)
