import pandas as pd
from collections import defaultdict
from math import log2
import re


def process_sequence():
    # 读取原始数据
    df = pd.read_csv("totalstrain.csv")

    # ===== 第一阶段：2-gram熵计算 =====
    class_stats = defaultdict(lambda: {'total': 0, 'grams': defaultdict(int)})

    # 统计每个类别的2-gram分布
    for _, row in df.iterrows():
        if not isinstance(row['text'], str):
            continue
        ops = row['text'].split()
        grams = list(zip(ops[:-1], ops[1:]))
        cls = row['Class']

        class_stats[cls]['total'] += len(grams)
        for gram in grams:
            class_stats[cls]['grams'][gram] += 1

    # 计算每个2-gram的熵
    all_grams = set()
    for stats in class_stats.values():
        all_grams.update(stats['grams'].keys())

    gram_entropy = {}
    for gram in all_grams:
        probs = []
        for cls, stats in class_stats.items():
            total = stats['total'] or 1  # 防止除零
            count = stats['grams'].get(gram, 0)
            probs.append(count / total)

        # 归一化概率
        total_p = sum(probs)
        if total_p == 0:
            continue
        norm_probs = [p / total_p for p in probs]

        # 计算熵值
        entropy = -sum(p * log2(p) for p in norm_probs if p > 0)
        gram_entropy[gram] = entropy

    # 选择熵值最高的510个2-gram
    top_grams = set([gram for gram, _ in sorted(
        gram_entropy.items(),
        key=lambda x: x[1],
        reverse=True
    )[:510]])

    # ===== 第二阶段：序列重构 =====
    def reconstruct_sequence(ops):
        """保留高熵2-gram的原始顺序"""
        if len(ops) < 2:
            return ' '.join(ops)

        # 标记需要保留的索引
        keep_indices = set()
        for i in range(len(ops) - 1):
            current_gram = (ops[i], ops[i + 1])
            if current_gram in top_grams:
                keep_indices.add(i)
                keep_indices.add(i + 1)

        # 按原始顺序过滤并保留
        filtered_ops = [ops[i] for i in sorted(keep_indices)]
        return ' '.join(filtered_ops) if filtered_ops else ''

    # 应用处理并保存
    df['texts'] = df['text'].apply(
        lambda x: reconstruct_sequence(x.split()) if isinstance(x, str) else ''
    )

    # 保留必要字段
    df[['Id', 'Class', 'texts']].to_csv("totaltrains-sequence.csv", index=False)
    print("处理完成，结果保存至 totaltrains-sequence.csv")


if __name__ == "__main__":
    process_sequence()