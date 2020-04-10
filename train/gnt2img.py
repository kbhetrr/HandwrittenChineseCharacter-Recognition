import pandas as pd
import os
from reader import Reader
import sys

# 读取汉字对应关系
table = pd.read_csv("database/gb2312_level1.csv")
value = table.values
ids = [item[4] for item in value]
chars = [item[2] for item in value]
id2char = dict(zip(ids, chars))
char2id = dict(zip(chars, ids))

# 创建每个汉字的对应目录
for char in chars:
    path = "database/test_56/"+char
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)


reader = Reader()
dirs = ['database/CASIA_HWDB1.1_TEST/']  # 数据集路径
data_list = []  # (image, label)数据对

for dir_path in dirs:  # 遍历文件夹中的所有gnt文件
    files = os.listdir(dir_path)
    length = len(files)
    for index, file in enumerate(files):
        file_path = dir_path + file
        data_list.extend(reader.read_gnt_image(file_path))  # gnt转png
        sys.stdout.write('\r>> Dealing gnt file %d/%d' % (index, length))
        sys.stdout.flush()

count = [0 for i in range(3755)]
length = len(data_list)
for index, (image, label) in enumerate(data_list):
    if label in chars:
        image.save("database/test_56/"+label+"/"+str(count[char2id[label]])+".png")
        count[char2id[label]] += 1
        sys.stdout.write('\r>> Dealing gnt file %d/%d' % (index, length))
        sys.stdout.flush()
    else:
        print(label)