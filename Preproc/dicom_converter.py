import os
import cv2
import glob
import shutil
import pydicom
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import mritopng

root_path        = "/home/fitogether/Documents/medical_image_liver_fibrosis/DataSet/"
train_dicom_path  = os.path.join(root_path, "train")
train_result_path = os.path.join(root_path, "train_whole")
test_dicom_path   = os.path.join(root_path, "test")
test_result_path  = os.path.join(root_path, "test_whole")
val_dicom_path   = os.path.join(root_path, "val")
val_result_path  = os.path.join(root_path, "val_whole")

# print("Converting Train Datasets ... ")
# os.chdir(train_dicom_path)
# files = glob.glob("*.dcm")
# for file in tqdm(files):
#     fname = file[:-4] + ".png"
#
#     data = pydicom.dcmread(file)
#     img  = data.pixel_array
#     plt.imsave(os.path.join(train_result_path, fname), img, cmap=plt.cm.bone)
#     # Image.save(os.path.join(train_result_path, fname))
#     # cv2.imwrite(os.path.join(train_result_path, fname), img)

# print("Converting Test Datasets ... ")
# os.chdir(test_dicom_path)
# files = glob.glob("*.dcm")
# for file in tqdm(files[:1]):
#     try:
#         fname = file[:-4] + ".png"
#
#         data = pydicom.dcmread(file)
#         img = data.pixel_array
#         print(img.shape)
#         plt.imsave(os.path.join(test_result_path, fname), img, cmap=plt.cm.bone)
#         # Image.save(os.path.join(test_result_path, fname))
#         # cv2.imwrite(os.path.join(test_result_path, fname), img)
#     except:
#         print(f"unexpected error occur: {file}")

print("Converting Test Datasets ... ")
os.chdir(val_dicom_path)
files = glob.glob("*.dcm")
for file in tqdm(files):
    try:
        fname = file[:-4] + ".png"

        mritopng.convert_file(os.path.join(val_dicom_path, file), os.path.join(val_result_path, fname))
        # data = pydicom.dcmread(file)
        # img = data.pixel_array
        # print(img.shape)
        # plt.imsave(os.path.join(test_result_path, fname), img, cmap=plt.cm.bone)
        # Image.save(os.path.join(test_result_path, fname))
        # cv2.imwrite(os.path.join(test_result_path, fname), img)
    except:
        print(f"unexpected error occur: {file}")