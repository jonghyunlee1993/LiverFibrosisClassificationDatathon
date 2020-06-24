import os
import glob
import shutil
import numpy as np
import pandas as pd

path = os.getcwd()

train_data = pd.read_csv(os.path.join(path, 'train.csv'))
# print(train_data.head())

train_path = os.path.join(path, 'DataSet', 'train_png')
f1_path = os.path.join(train_path, 'F1')
f2_path = os.path.join(train_path, 'F2')
f3_path = os.path.join(train_path, 'F3')

train_data['fname'] = train_data.ID.apply(lambda x: x[3:])
for data in train_data.iterrows():
    try:
        label   = int(data[1][1])
        file_id = str(data[1][2]) + '.dcm.png'
        file_id_rename = str(data[1][2]) + '.png'

        # os.rename(os.path.join(train_path, file_id), os.path.join(train_path, file_id_rename))
        src = os.path.join(train_path, file_id_rename)

        if label == 1:
            des = os.path.join(f1_path, file_id_rename)
        elif label == 2:
            des = os.path.join(f2_path, file_id_rename)
        elif label == 3:
            des = os.path.join(f3_path, file_id_rename)

        shutil.move(src, des)
    except:
        print(f"File dose not exist {file_id_rename}")