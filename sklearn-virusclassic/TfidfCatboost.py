import pandas as pd
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer          #For Bag of words
from sklearn.feature_extraction.text import TfidfVectorizer          #For TF-IDF
# from sklearn.model_selection import train_test_split
# from gensim.models import Word2Vec                                   #For Word2Vec
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


data_folder = "E:\pycharmcode\sklearn-virusclassic\subtrain_with_text.csv"
df = pd.read_csv(data_folder)
# label
label = df['Class']
# data
data = df['text']
# 填充缺失值
data.fillna("", inplace=True)

x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)

######### TfidfVectorizer
tf_idf = TfidfVectorizer(max_features=100, lowercase=True)
# 拟合和转换训练数据
x_train_trans = tf_idf.fit_transform(x_train)
# 根据在训练数据上拟合得到的词典计算和转换测试数据
x_test_trans = tf_idf.transform(x_test)

clf = CatBoostClassifier(iterations=1000, learning_rate=0.001, loss_function='MultiClass', verbose=True,
                         random_seed=42)  # verbose显示训练进度
clf.fit(x_train_trans, y_train)
predictions = clf.predict(x_test_trans)

# accuracy = sum(predictions.flatten() == y_test) / len(y_test)
# 计算精度
accuracy = accuracy_score(predictions, y_test)
print(f"Accuracy: {accuracy}")  # Accuracy: 0.8881987577639752
