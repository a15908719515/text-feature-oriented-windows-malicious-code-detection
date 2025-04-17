import pandas as pd
import numpy as np
from collections import Counter
from math import log2
import seaborn as sns
import matplotlib.pyplot as plt
import re


# --------------------- 第一阶段：分块处理 ---------------------
def split_blocks(text):
    """分割操作码序列为块，并标记是否包含call指令"""
    if not isinstance(text, str) or not text.strip():
        return []

    ops = text.split()
    blocks = []
    current_block = []
    has_call = False

    for op in ops:
        current_block.append(op)
        # 判断是否为分隔符（原逻辑）
        if op.lower() == "call" or op.lower().startswith("ret") or re.match(r"^j[a-z]{1,3}$", op.lower()):
            # 记录是否包含call
            block_has_call = "call" in [x.lower() for x in current_block]
            blocks.append({
                "tokens": current_block,
                "has_call": block_has_call,
                "length": len(current_block)
            })
            current_block = []
            has_call = False

    # 处理最后一个块
    if current_block:
        block_has_call = "call" in [x.lower() for x in current_block]
        blocks.append({
            "tokens": current_block,
            "has_call": block_has_call,
            "length": len(current_block)
        })

    return blocks


# --------------------- 第二阶段：信息熵计算 ---------------------
def calculate_entropy(block):
    """计算单个块的信息熵"""
    if len(block["tokens"]) == 0:
        return 0.0

    # 统计操作码频率
    counter = Counter(block["tokens"])
    total = len(block["tokens"])

    entropy = 0.0
    for count in counter.values():
        p = count / total
        entropy -= p * log2(p)

    return entropy


def analyze_blocks(df):
    """执行完整的熵分析流程"""
    # 展开所有块
    expanded_data = []
    for _, row in df.iterrows():
        blocks = split_blocks(row["text"])
        for block in blocks:
            expanded_data.append({
                "class": row["Class"],
                "has_call": block["has_call"],
                "entropy": calculate_entropy(block),
                "length": block["length"]
            })

    analysis_df = pd.DataFrame(expanded_data)

    # --------------------- 分析结果输出 ---------------------
    print("\n【熵值对比分析】")
    call_blocks = analysis_df[analysis_df["has_call"]]
    non_call_blocks = analysis_df[~analysis_df["has_call"]]

    print(f"包含call的块数量: {len(call_blocks):,}")
    print(f"平均熵值: {call_blocks['entropy'].mean():.3f} ± {call_blocks['entropy'].std():.3f}")
    print(f"平均长度: {call_blocks['length'].mean():.1f} tokens")

    print(f"\n不包含call的块数量: {len(non_call_blocks):,}")
    print(f"平均熵值: {non_call_blocks['entropy'].mean():.3f} ± {non_call_blocks['entropy'].std():.3f}")
    print(f"平均长度: {non_call_blocks['length'].mean():.1f} tokens")

    class_analysis = analysis_df.groupby(["class", "has_call"])["entropy"].agg(["mean", "std"])
    print("\n按家族分类的熵值分析:")
    print(class_analysis)

    # 执行T检验
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(
        call_blocks["entropy"],
        non_call_blocks["entropy"],
        equal_var=False
    )
    print(f"\nT检验结果: t={t_stat:.2f}, p={p_value:.4f}")

    # --------------------- 可视化展示 ---------------------
    plt.figure(figsize=(10, 6))
    sns.violinplot(
        x="has_call",
        y="entropy",
        data=analysis_df,
        palette="Set2"
    )
    plt.title("Call块与非Call块的信息熵分布对比")
    plt.xlabel("包含Call指令")
    plt.ylabel("信息熵值")
    plt.xticks([0, 1], ["非Call块", "Call块"])
    plt.savefig("entropy_comparison.png", dpi=300)
    plt.show()


# --------------------- 主执行流程 ---------------------
if __name__ == "__main__":
    # 读取数据
    df = pd.read_csv("totalstrain.csv")

    # 执行分析
    analyze_blocks(df)