import csv
from random import random

import pandas as pd
df = pd.read_csv("train.csv")
labels = df['Class']
contents = df['text']
count = {}
cal = {}
for p in df['Class']:
    cal[p] = 1
    try:
        count[p] += 1
    except KeyError:
        count[p] = 1
print(count)
"""输出如下
{
	'SCIENCE': 3774, 'TECHNOLOGY': 15000,
	'HEALTH': 15000, 'WORLD': 15000,
	'ENTERTAINMENT': 15000, 'SPORTS': 15000,
	'BUSINESS': 15000, 'NATION': 15000
}
"""



