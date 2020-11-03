'''
read picture names, file_dir is the main dir of data
'''
import os
from PIL import Image
'''
get the max height pix and width pix from the whole dataset
'''


file_dir = "retrain"
for root, dirs, files in os.walk(file_dir, topdown=False):
    print(root)# 当前目录路径
    if root == file_dir:
        continue
    isExists = os.path.exists(os.path.join("retrain2", root.split("/")[1]))
    if not isExists:
        os.makedirs(os.path.join("retrain2", root.split("/")[1]))
    else:
        ls = os.listdir(os.path.join("retrain2", root.split("/")[1]))
        for i in ls:
            f_path = os.path.join("retrain2/" + root.split("/")[1], i)
            os.remove(f_path)
    print(files)  # 当前路径下所有非目录子文件
    for i in files:
        print(os.path.join(root, i))
        pic = os.path.join(root, i)
        img = Image.open(pic)
        print(pic)
        pic = img.resize((100, 100))
        pic.save(os.path.join("retrain2/" + root.split("/")[1], i))


