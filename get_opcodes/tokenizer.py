import torch
from transformers import BertTokenizer
from transformers import RobertaTokenizer, RobertaModel



def prepare_bert_input(opcode_txt_path, model_name):
    # 1. 加载操作码序列
    with open(opcode_txt_path, "r") as f:
        opcodes = f.read().split()

    # 2. 初始化分词器
    tokenizer = BertTokenizer.from_pretrained(model_name)
    tokenizer.add_tokens(["push", "mov", "call", "jmp"])  # 按需扩展

    # 3. 分块处理超长序列
    max_seq_length = 512
    chunks = []
    for i in range(0, len(opcodes), max_seq_length - 2):  # 保留[CLS]/[SEP]
        chunk = opcodes[i:i + max_seq_length - 2]
        chunks.append(chunk)

    # 4. 转换为模型输入
    batch_inputs = []
    for chunk in chunks:
        inputs = tokenizer(
            chunk,
            add_special_tokens=True,
            max_length=max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        batch_inputs.append(inputs)

    # 5. 合并分块（可选，根据模型需求）
    if len(batch_inputs) == 1:
        return batch_inputs[0]
    else:
        return {
            "input_ids": torch.cat([x["input_ids"] for x in batch_inputs], dim=0),
            "attention_mask": torch.cat([x["attention_mask"] for x in batch_inputs], dim=0)
        }


# 使用示例
model_name= 'E:\pycharmcode\codebert-base'
opcode_txt_path = "asm.txt"
bert_inputs = prepare_bert_input(opcode_txt_path,model_name)
print("Processed Input Shape:", bert_inputs["input_ids"].shape)