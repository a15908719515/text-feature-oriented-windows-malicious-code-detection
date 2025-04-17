# ## 创建英文词典
import operator
import config
import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle
import os
import pickle as pkl
import torch
tqdm.pandas(desc='pandas bar')

MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


# ## 去除特殊字符
def clean_special_chars(text, punct, mapping):
    for p in mapping:
        text = text.replace(p, mapping[p])
    for p in punct:
        text = text.replace(p, f' {p} ')
    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  # Other special characters that I have to deal with in last
    for s in specials:
        text = text.replace(s, specials[s])
    return text


def build_vocab(file_path, max_size, min_freq):
    df = pd.read_csv(file_path)
    # 转化为小写
    sentences = df['text'].apply(lambda x: x.lower())
    # 去除特殊字符
    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
    punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2",
                     "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e",
                     '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta',
                     '∅': '', '³': '3', 'π': 'pi', }
    sentences = sentences.apply(lambda x: clean_special_chars(x, punct, punct_mapping))
    # 提取数组
    sentences = sentences.progress_apply(lambda x: x.split()).values
    vocab_dic = {}
    for sentence in tqdm(sentences, disable=False):
        for word in sentence:
            try:
                vocab_dic[word] += 1
            except KeyError:
                vocab_dic[word] = 1
    vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
    vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
    vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


def build_dataset(config):
    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        vocab = build_vocab(config.train_path, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))
    print(f"词典======== {vocab}")

if __name__ == "__main__":
    class Config():
        def __init__(self):
            self.vocab_path = 'vocab.pkl'
            self.train_path = 'train.csv'
            self.dev_path = 'dev.csv'
            self.test_path = 'test.csv'
            self.pad_size = 14

    build_dataset(Config())


# ## 加载预训练词向量
def load_embed(file):
    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    if file == 'wiki-news-300d-1M.vec':
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file, encoding='utf-8') if len(o) > 100)
    else:
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file, encoding='latin'))

    return embeddings_index


ebed = []

def get_embed(vocab_path, embed_path,dim):
    vocab = pkl.load(open(vocab_path, 'rb'))
    embed_glove = load_embed(embed_path)
    for v in vocab:
        if v not in embed_glove.keys():
            ebed.append(np.asarray([0 for i in range(0,dim)], dtype='float32'))
        else:
            ebed.append(embed_glove[v])
    return np.asarray(ebed, dtype='float32')


vocab_path = 'vocab.pkl'
embed_path = 'glove.6B.300d.txt'
dim = 300

np.savez('glove.6B.300d.npz',embeddings=get_embed(vocab_path, embed_path, dim))


#建立数据集dataset
def build_dataset(config):
    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        vocab = build_vocab(config.train_path, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))
    print(f"词典大小======== {len(vocab)}")

    def load_dataset(path, pad_size=32):
        df = pd.read_csv(path)
        # TODO 这里读数据集写死了 title
        # 转化为小写
        sentences = df['text'].apply(lambda x: x.lower())
        # 去除特殊字符
        punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
        punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2",
                         "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e",
                         '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-',
                         'β': 'beta',
                         '∅': '', '³': '3', 'π': 'pi', }
        sentences = sentences.apply(lambda x: clean_special_chars(x, punct, punct_mapping))
        # 提取数组
        sentences = sentences.progress_apply(lambda x: x.split()).values
        labels = df['Class']
        labels_id = list(set(df['Class']))
        labels_id.sort()
        contents = []
        count = 0
        for i, token in tqdm(enumerate(sentences)):
            label = labels[i]
            words_line = []
            seq_len = len(token)
            count += seq_len
            if pad_size:
                if len(token) < pad_size:
                    token.extend([PAD] * (pad_size - len(token)))
                else:
                    token = token[:pad_size]
                    seq_len = pad_size
            # word to id
            for word in token:
                words_line.append(vocab.get(word, vocab.get(UNK)))
            contents.append((words_line, labels_id.index(label), seq_len))
        print(f"数据集地址========{path}")
        print(f"数据集总词数========{count}")
        print(f"数据集文本数========{len(sentences)}")
        print(f"数据集文本平均词数========{count/len(sentences)}")
        print(f"训练集标签========{set(df['label'])}")
        return contents  # [([...], 0), ([...], 1), ...]
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)

    return vocab, train, dev, test

vocab, train_data, dev_data, test_data = build_dataset(config)

class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter

train_iter = build_iterator(train_data, config)
dev_iter = build_iterator(dev_data, config)
test_iter = build_iterator(test_data, config)






