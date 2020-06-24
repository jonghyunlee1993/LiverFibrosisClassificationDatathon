import os
import glob
import shutil
import numpy as np
import pandas as pd

path = os.getcwd()

test_data = pd.read_csv(os.path.join(path, 'test.csv'))
# print(train_data.head())

test_path = os.path.join(path, 'DataSet', 'test_whole')
f1_path = os.path.join(test_path, 'F1')
f2_path = os.path.join(test_path, 'F2')
f3_path = os.path.join(test_path, 'F3')

test_data['fname'] = test_data.ID.apply(lambda x: x[3:])
for data in test_data.iterrows():
    try:
        label   = int(data[1][1])
        file_id = str(data[1][2]) + '.dcm.png'
        file_id_rename = str(data[1][2]) + '.png'

        # os.rename(os.path.join(test_path, file_id), os.path.join(test_path, file_id_rename))
        src = os.path.join(test_path, file_id_rename)

        if label == 1:
            des = os.path.join(f1_path, file_id_rename)
        elif label == 2:
            des = os.path.join(f2_path, file_id_rename)
        elif label == 3:
            des = os.path.join(f3_path, file_id_rename)

        shutil.move(src, des)
    except:
        print(f"Error {data}")
