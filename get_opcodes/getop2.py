import re
from collections import *
import os
import pandas as pd
import numpy as np
import csv


def getOpcodeSequence(filename):
    # 获取操作码序列
    opcode_seq = []
    # p = re.compile(r'\s([a-fA-F0-9]{2}\s)+\s*([a-z]+)')
    p = re.compile(r'\s+((?:[a-fA-F0-9]{2}\s*)+)\s+([a-z]+)')
    with open(filename, mode="r", encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith(".text"):
                m = re.findall(p, line)
                # print(m)
                if m:
                    opc = m[0][1]
                    if opc != "align":
                        opcode_seq.append(opc)
    return opcode_seq


# ... 前面的代码保持不变 ...

# path = r"E:\pycharmcode\get_opcodes"  # 文件夹目录
# files = os.listdir(path)  # 得到文件夹下的所有文件名称
# i = 0
# temp = []
# test = []
# for file in files:  # 遍历文件夹
#     filename = path + '\\' + file
#     print(filename)
#     i = i + 1
#     print(i)
#     temp = getOpcodeSequence(filename)
#     if temp == test:
#         continue
#     temp.insert(0, file)
#
#     # 修改后的TXT保存部分
#     with open(r'E:\pycharmcode\get_opcodes\asm.txt', 'a', encoding='UTF8') as f:
#         line = ' '.join(temp)  # 用空格连接列表元素
#         f.write(line + '\n')  # 写入行并换行

path = r"E:\pycharmcode\get_opcodes"  # 文件夹目录
files = os.listdir(path)  # 得到文件夹下的所有文件名称
i = 0
temp = []
test = []
for file in files:  # 遍历文件夹
    filename = path + '\\' + file
    print(filename)
    i = i + 1
    print(i)
    temp = getOpcodeSequence(filename)
    if temp == test:
        continue
    temp.insert(0, file)
    # df = pd.DataFrame(temp)
    # print(df)
    # df.append(temp)
    # df.to_csv(r"E:\pycharmcode\get_opcodes\asm.csv",mode = 'a',index =False)
    with open(r'E:\pycharmcode\get_opcodes\asm.csv', 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        # 写入数据
        writer.writerow(temp)
