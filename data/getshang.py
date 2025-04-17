import pandas as pd
from collections import defaultdict
from math import log2


def process_dataset():
    # 读取原始数据
    df = pd.read_csv("totalstrain.csv")

    # 统计每个样本的2-gram特征
    def get_2grams(text):
        if not isinstance(text, str) or len(text.strip()) < 2:
            return []
        ops = text.split()
        return list(zip(ops[:-1], ops[1:]))

    # 第一阶段：统计全局2-gram信息
    class_stats = defaultdict(lambda: {
        'total_grams': 0,
        'gram_counts': defaultdict(int)
    })

    # 遍历所有样本建立统计信息
    for _, row in df.iterrows():
        grams = get_2grams(row['text'])
        class_label = row['Class']

        # 更新类别统计
        class_stats[class_label]['total_grams'] += len(grams)
        for gram in grams:
            class_stats[class_label]['gram_counts'][gram] += 1

    # 第二阶段：计算信息熵
    gram_entropy = {}
    all_grams = set()

    # 收集所有存在的2-gram
    for stats in class_stats.values():
        all_grams.update(stats['gram_counts'].keys())

    # 计算每个2-gram的熵
    for gram in all_grams:
        probabilities = []
        for class_label, stats in class_stats.items():
            total = stats['total_grams'] or 1  # 避免除零
            count = stats['gram_counts'].get(gram, 0)
            probabilities.append(count / total)

        # 归一化概率
        total_p = sum(probabilities)
        if total_p == 0:
            continue
        norm_probs = [p / total_p for p in probabilities]

        # 计算信息熵
        entropy = -sum(p * log2(p) for p in norm_probs if p > 0)
        gram_entropy[gram] = entropy

    # 选择熵值最高的600个特征
    top_grams = set([gram for gram, _ in sorted(
        gram_entropy.items(),
        key=lambda x: x[1],
        reverse=True
    )[:600]])

    # 第三阶段：生成新序列
    def filter_sequence(text):
        if not isinstance(text, str):
            return ""

        ops = text.split()
        keep_indices = set()

        # 标记需要保留的索引
        for i in range(len(ops) - 1):
            if (ops[i], ops[i + 1]) in top_grams:
                keep_indices.add(i)
                keep_indices.add(i + 1)

        # 合并连续区间
        sorted_indices = sorted(keep_indices)
        filtered_ops = []
        last = -2

        for idx in sorted_indices:
            if idx > last + 1:  # 新区间开始
                if filtered_ops:
                    filtered_ops.append("...")  # 非连续部分标记
                filtered_ops.append(ops[idx])
            else:  # 连续部分
                filtered_ops.append(ops[idx])
            last = idx

        return " ".join(filtered_ops) if filtered_ops else text

    # 生成新列并保存
    df['texts'] = df['text'].apply(filter_sequence)
    df[['Id', 'Class', 'texts']].to_csv("totaltrains.csv", index=False)
    print("处理完成，结果已保存到 totaltrains.csv")


if __name__ == "__main__":
    process_dataset()