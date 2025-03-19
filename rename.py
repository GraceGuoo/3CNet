import re
import os,shutil                                            #导入模块
#rename
# path = "/root/dataset/mfnet/labels/"
# path2 = "/root/dataset/rgbx_mfnet/Labels/"
# for filename in os.listdir(path):
#     filename = str(filename)#遍历列表下的所有文件名
#     # if filename.find("th")>0:            #当文件名不为“001.txt”时
#     if filename.find("png")>0:            #当文件名不为“001.txt”时
#         new_name=filename       #为文件赋予新名字
#         # new_name=filename.replace('_th','')               #为文件赋予新名字
#         shutil.copyfile(path+filename, path2+new_name)    #复制并重命名文件
#         print(filename,"copied as",new_name)
#     # copy_files(path, path2)         #调用定义的函数，注意名称与定义的函数名一致

path = "/root/dataset/rgbx_mfnet/"
filename1 = path+"train.txt"
filename2 = path+"train_new.txt"
l1 = []
with open(filename1,'r') as f1:
    l1 = f1.readlines()
with open(filename2,'w') as f2:
    for line in l1:
        if line.find("flip")<0:
            f2.write(line)
