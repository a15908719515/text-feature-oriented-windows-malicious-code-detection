from transformers import BertModel,BertTokenizer

BERT_PATH = "E:\pycharmcode\\bert-base-uncased"

tokenizer = BertTokenizer.from_pretrained(BERT_PATH)

print(tokenizer.tokenize('I have a good time, thank you.'))

bert = BertModel.from_pretrained(BERT_PATH)

print('load bert model over')
