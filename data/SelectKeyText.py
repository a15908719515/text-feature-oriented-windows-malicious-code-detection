import pandas as pd
import re


def is_separator(op):
    """判断是否为分隔符（跳转/调用/返回指令）"""
    op_lower = op.lower()
    return (
            op_lower == "call" or
            op_lower.startswith("ret") or
            bool(re.match(r"^j[a-z]{1,3}$", op_lower))
    )


def extract_key_blocks(text):
    """提取关键块，新增长度判断和空结果处理"""
    # 处理空值和非字符串
    if not isinstance(text, str) or not text.strip():
        return text

    original_ops = text.split()

    # 条件1：原序列长度小于600时不处理
    if len(original_ops) < 600:
        return text

    # 执行分块处理
    blocks = []
    current_block = []
    for op in original_ops:
        current_block.append(op)
        if is_separator(op):
            blocks.append(current_block)
            current_block = []
    if current_block:
        blocks.append(current_block)

    # 筛选含call的块
    key_ops = []
    for block in blocks:
        if any(op.lower() == "call" for op in block):
            key_ops.extend(block)

    processed_text = " ".join(key_ops)

    # 条件2：处理结果为空时返回原序列
    return processed_text if processed_text.strip() else text


# 读取并处理数据
df = pd.read_csv("totalstrain.csv")
df["keytext"] = df["text"].apply(extract_key_blocks)

# 保留结果列
df[["Id", "Class", "keytext"]].to_csv("totalstrainkey.csv", index=False)

print("处理完成，新增条件已生效")