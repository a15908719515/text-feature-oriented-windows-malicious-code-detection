# ## 创建英文词典
import operator
import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle

def build_vocab(sentences, verbose=True):
    vocab = {}
    for sentence in tqdm(sentences, disable=(not verbose)):
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab

# ## 进度条初始化
tqdm.pandas()
# ## 加载数据集
df = pd.read_csv("train.csv")
# ## 创建词典
sentences = df['text'].progress_apply(lambda x: x.split()).values
vocab = build_vocab(sentences)
# ## 加载预训练词向量
def load_embed(file):
    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    if file == 'wiki-news-300d-1M.vec':
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file,encoding='utf-8') if len(o) > 100)
    else:
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file, encoding='latin'))

    return embeddings_index

# ## 加载词向量
glove = 'glove.6B.50d.txt'
fasttext = 'wiki-news-300d-1M.vec'
embed_glove = load_embed(glove)
embed_fasttext = load_embed(fasttext)


# ## 检查预训练embeddings和vocab的覆盖情况
def check_coverage(vocab, embeddings_index):
    known_words = {}  # 两者都有的单词
    unknown_words = {}  # embeddings不能覆盖的单词
    nb_known_words = 0  # 对应的数量
    nb_unknown_words = 0
    #     for word in vocab.keys():
    for word in tqdm(vocab):
        try:
            known_words[word] = embeddings_index[word]
            nb_known_words += vocab[word]
        except:
            unknown_words[word] = vocab[word]
            nb_unknown_words += vocab[word]
            pass

    print('Found embeddings for {:.2%} of vocab'.format(len(known_words) / len(vocab)))  # 覆盖单词的百分比
    print('Found embeddings for  {:.2%} of all text'.format(
        nb_known_words / (nb_known_words + nb_unknown_words)))  # 覆盖文本的百分比，与上一个指标的区别的原因在于单词在文本中是重复出现的。
    unknown_words = sorted(unknown_words.items(), key=operator.itemgetter(1))[::-1]
    print("unknown words : ", unknown_words[:30])
    return unknown_words

oov_glove = check_coverage(vocab, embed_glove)
oov_fasttext = check_coverage(vocab, embed_fasttext)









