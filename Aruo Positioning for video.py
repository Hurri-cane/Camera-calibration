#*********************************************************************#
#*                           By Huang Wenjun                         *#
#*********************************************************************#
import cv2 as cv
import cv2.aruco as aruco
import  numpy as np
from math import degrees as dg
import math



def cv_show(frame,name='Figure'):
    cv.namedWindow(name,cv.WINDOW_AUTOSIZE)
    cv.imshow(name,frame)
    cv.waitKey(0)
    cv.destroyAllWindows()

def isRotationMatrix(R):
    Rt = np.transpose(R)  # 旋转矩阵R的转置
    shouldBeIdentity = np.dot(Rt, R)  # R的转置矩阵乘以R
    I = np.identity(3, dtype=R.dtype)  # 3阶单位矩阵
    n = np.linalg.norm(I - shouldBeIdentity)  # np.linalg.norm默认求二范数
    return n < 1e-6  # 目的是判断矩阵R是否正交矩阵（旋转矩阵按道理须为正交矩阵，如此其返回值理论为0）


def rotationMatrixToAngles(R):
    assert (isRotationMatrix(R))  # 判断是否是旋转矩阵（用到正交矩阵特性）

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[
        1, 0])  # 矩阵元素下标都从0开始（对应公式中是sqrt(r11*r11+r21*r21)），sy=sqrt(cosβ*cosβ)

    singular = sy < 1e-6  # 判断β是否为正负90°

    if not singular:  # β不是正负90°
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:  # β是正负90°
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)  # 当z=0时，此公式也OK，上面图片中的公式也是OK的
        z = 0

    return np.array([x, y, z])

aruco_size = 0.076

with np.load(r'F:\PyCharm\Camera_calibration_GIT\class1\class_mtx.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]
print("#######加载相机内参和畸变矩阵#######")
print(mtx, '\n\n', dist)

aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters_create()

index = 0
cap = cv.VideoCapture(0,cv.CAP_DSHOW)  #更改API设置
flag = cap.isOpened()
cap.set(3, 1280)
cap.set(4, 720)

Camear_Point = None
rvec = None
tvec = None
point1 = None
point2 = None
Distance_1 = []
Distance_2 = []
Rz = []
# zero = np.array([[0, 0, 0]], float).T
# mtx_comb = np.concatenate((mtx, zero), axis=1)
# mtx_comb = np.mat(mtx_comb)
# dst_comb, _ = cv.Rodrigues(dist)
# translation_vec = np.array([[0], [0], [0]])
# Tr  = np.concatenate((dst_comb, translation_vec), axis=1)
# full = np.array([[0,0,0,1]])
# Tr  = np.concatenate((Tr, full), axis=0)
# inparam_outparam = mtx_comb*Tr
# left3 = inparam_outparam[:,:3]
# right1 = inparam_outparam[:,3:]



while (flag):
    ret, frame = cap.read()
    frame_copy = frame.copy()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    corners, ids, rejectedframePoints = aruco.detectMarkers(frame_gray, aruco_dict, parameters=parameters)
    font = cv.FONT_HERSHEY_SIMPLEX

    if ids is not None:
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, aruco_size, mtx, dist)  # aruco码边长为0.028m
        (rvec - tvec).any()

        for i in range(rvec.shape[0]):
            aruco.drawAxis(frame, mtx, dist, rvec[i, :, :], tvec[i, :, :], aruco_size)
            aruco.drawDetectedMarkers(frame, corners, ids, (0, 179, 255))
        cv.putText(frame, "Id: " + str(ids.T), (50, 80), font, 2, (0, 0, 255), 2, cv.LINE_AA)

        markerCenter = []
        sita_z = None
    ###计算位置
        for j in range(len(rvec)):



            # 方法一：采用aruco码的平移矩阵和旋转矩阵计算
            ##此处采用的是相机在Marker坐标系的坐标
            # dst, _ = cv.Rodrigues(rvec[j])
            # dst = np.mat(dst)
            # tr = tvec[j]
            # tr = np.mat(tr)
            # Camear_Point = -dst.I * tr.T
            # Camear_Point = Camear_Point * 1000  # 单位转换为mm
            ##此处采用的是相机在Marker坐标系的坐标

            #采用Marker在相机坐标系的坐标
            Xc = tvec[j][0][0]*1000
            Yc = tvec[j][0][1]*1000
            Zc = tvec[j][0][2]*1000
            Xc = int(Xc)
            Yc = int(Yc)
            Zc = int(Zc)


            print('\n', "-------------------------")
            print("Id: " + str(ids[j]) + '坐标为：')
            print('Xc=', Xc, 'mm', 'Yc=', Yc, 'mm', 'Zc=', Zc, 'mm')
            location1 = tuple(corners[j][0][0])
            # b = (200,200)
            # print(a)
            # cv.putText(frame, "Id: " + str(ids[j]) + 'solution1'+str([Xc,Yc,Zc]), location1, font, 1, (0, 255, 0), 2, cv.LINE_AA)


            # 方法二：采用aruco码的平corners的点计算
            uv_point = corners[j][0][2]
            # answer = uv_point - right1
            # Xw, Yw, Zw = left3.I * answer
            # Zc = 1000
            Xw = Zc*(uv_point[0]-mtx[0][2])/mtx[0][0]
            Yw = Zc*(uv_point[1]-mtx[1][2])/mtx[1][1]
            Zw = 1745
            location2 = tuple(corners[j][0][2])
            # cv.putText(frame, "Id: "+ str(ids[j])+ 'solution2' + str([int(Xw), int(Yw), int(Zw)]), location2, font, 1, (255, 0, 0), 2, cv.LINE_AA)

            # 计算转角：
            R, _ = cv.Rodrigues(rvec[j])
            [Rx,Ry,Rz]=rotationMatrixToAngles(R)
            sita_x = dg(Rx)
            sita_y = dg(Ry)
            sita_z = dg(Rz)
            sita_x = np.round(sita_x, 2)
            sita_y = np.round(sita_y, 2)
            sita_z = np.round(sita_z, 2)
            print('Rx=',sita_x,'Ry=',sita_y,'Rz=',sita_z)
            location3 = tuple(corners[j][0][1])
            # cv.putText(frame,  'Rotation' + str(sita_z), location3, font, 1,(0, 0, 255), 2, cv.LINE_AA)

            #仅存在两个aruco码时，进行距离计算
            if len(rvec) == 2:
                #方法一：采用aruco码的平移矩阵和旋转矩阵计算（仅含有两个标识）
                if j == 0:
                    point1 = np.mat([Xc ,Yc ,Zc])
                if j == 1:
                    point2 = np.mat([Xc ,Yc ,Zc])
                    distance_Mt = np.sqrt((point1-point2)*((point1-point2).T))
                    distance_Mt = int(distance_Mt)
                    Distance_1.append(distance_Mt)
                    cv.putText(frame, "Distance: "+str(distance_Mt), (50, 200), font, 1, (0, 0, 255), 2, cv.LINE_AA)


                #方法二：采用aruco码的平corners的点计算（仅含有两个标识）
                    #取中点
                    for k in range(len(ids)):
                        markerCenter.append(corners[k].sum(1) / 4.0)
                    C_point1 = markerCenter[0]
                    C_point2 = markerCenter[1]
                    print(C_point1, C_point2)
                    distance_pix = np.linalg.norm(C_point1 - C_point2)  # 求两点欧氏距离
                    # 取一个角点
                    # C_point1 = corners[0][0][2]
                    # C_point2 = corners[1][0][2]
                    # print(C_point1,C_point2)
                    # distance_pix = np.sqrt((C_point1[0]-C_point2[0])**2+(C_point1[1]-C_point2[1])**2)
                    distance_Co = distance_pix*Zc/mtx[0][0]
                    distance_Co = int(distance_Co)
                    Distance_2.append(distance_Co)
                    cv.putText(frame, "Corners distance: " + str(distance_Co), (50, 300), font, 1, (0, 0, 255), 2, cv.LINE_AA)



    else:
        cv.putText(frame, "No Ids", (50, 80), font, 2, (0, 0, 255), 3, cv.LINE_AA)

    cv.imshow("Capture_video", frame)
    k = cv.waitKey(1) & 0xFF
    if k == ord('s'):  # 按下s键，进入下面的保存图片操作
        cv.imwrite(r"F:\PyCharm\Camera_calibration_GIT\class5\0" + str(index) + ".jpg", frame_copy)
        print(cap.get(3))
        print(cap.get(4))
        print("save" + str(index) + ".jpg successfuly!")
        print("-------------------------")
        index += 1
    elif k == ord('q'):  # 按下q键，程序退出
        print('数据数：',len(Distance_1),'solution1:',np.mean(Distance_1),'solution2:',np.mean(Distance_2))
        print('标准差1：',np.std(Distance_1),'标准差2：',np.std(Distance_2))
        break
cap.release()
cv.destroyAllWindows()





