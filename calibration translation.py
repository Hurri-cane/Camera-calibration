# # --------------------------------------------------------
# #坐标转换

import cv2 as cv
import  numpy as np

with np.load(r'F:\PyCharm\Camera calibration\class1\class_mtx.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]
zero = np.array([[0,0,0]],float).T
mtx = np.concatenate((mtx, zero), axis=1)
mtx = np.mat(mtx)

print("#######加载相机内参和畸变矩阵#######")
print(mtx, '\n\n', dist)

Zc = 1000
uv_point = np.mat([0, 0, 1]).T*Zc
print('像素坐标系坐标，已经乘上Zc：\n',uv_point.T)

rotation_vec = np.mat([[-0.03785969], [-0.14613237], [-0.03201819]])
translation_vec = np.array([[0], [0], [0]])

#旋转向量转换为旋转矩阵
dst, _ = cv.Rodrigues(rotation_vec)
print('旋转矩阵:\n',dst)

Tr  = np.concatenate((dst, translation_vec), axis=1)

full = np.array([[0,0,0,1]])
Tr  = np.concatenate((Tr, full), axis=0)
print('世界坐标系到相机坐标系的变换矩阵:\n',Tr)

inparam_outparam = mtx*Tr
print('内参*外参：\n',inparam_outparam)

left3 = inparam_outparam[:,:3]
right1 = inparam_outparam[:,3:]
answer = uv_point-right1
Xw,Yw,Zw = left3.I*answer

print('pix_to_word\n','Xw=',Xw,'Yw=',Yw,'Zw=',Zw)

#验证世界坐标
Word_point = np.mat([200,300,1000,1]).T
u,v,z = inparam_outparam*Word_point/Zc
print('word_to_pix\n','u=',u,'v=',v,'zc=',z)
