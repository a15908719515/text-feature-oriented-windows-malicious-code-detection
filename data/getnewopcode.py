import pandas as pd
from collections import Counter


def process_opcodes(text):
    """处理操作码序列的核心函数"""
    if pd.isna(text) or not str(text).strip():
        return ""

    # 分割原始操作码序列
    opcodes = str(text).strip().split()
    original_len = len(opcodes)

    # 第一阶段处理：删除top3高频操作码（当长度>512时）
    if original_len > 512:
        counter = Counter(opcodes)
        # 获取频率最高的3个操作码（考虑并列情况）
        top3 = [item[0] for item in counter.most_common(3)]
        # 过滤掉这些操作码的所有出现
        opcodes = [op for op in opcodes if op not in top3]


    if original_len > 512:
        counter = Counter(opcodes)
        # 获取频率最高的3个操作码（考虑并列情况）
        top3 = [item[0] for item in counter.most_common(3)]
        # 过滤掉这些操作码的所有出现
        opcodes = [op for op in opcodes if op not in top3]


    # 第二阶段处理：复制单次出现的操作码
    if opcodes:  # 防止处理空列表
        new_counter = Counter(opcodes)
        new_sequence = []
        for op in opcodes:
            if new_counter[op] == 1:
                new_sequence.extend([op] * 3)  # 复制3次
            else:
                new_sequence.append(op)
        return " ".join(new_sequence)
    return ""


# 读取合并后的数据集
df = pd.read_csv("totalstrain.csv")

# 应用处理函数
df["newopcode"] = df["text"].apply(process_opcodes)

# 保存结果（保留原始ID和Class）
df[["Id", "Class", "newopcode"]].to_csv("strain_newopcode.csv", index=False)