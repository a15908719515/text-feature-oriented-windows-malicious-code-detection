from transformers import RobertaTokenizer, RobertaModel

# 指定本地模型路径
model_path = "E:\pycharmcode\codebert-base"

# 加载分词器和模型
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaModel.from_pretrained(model_path)

print("模型加载成功！")