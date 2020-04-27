#*********************************************************************#
#*                           By Huang Wenjun                         *#
#*********************************************************************#
import cv2 as cv
import cv2.aruco as aruco
import  numpy as np


def cv_show(img,name='Figure'):
    cv.namedWindow(name,cv.WINDOW_AUTOSIZE)
    cv.imshow(name,img)
    cv.waitKey(0)
    cv.destroyAllWindows()

aruco_size = 0.076

with np.load(r'F:\PyCharm\Camera_calibration_GIT\class1\class_mtx.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]



print("#######加载相机内参和畸变矩阵#######")
print(mtx, '\n\n', dist)

img = cv.imread(r'F:\PyCharm\Camera_calibration_GIT\class4\07.jpg')
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv_show(img)

aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters_create()
corners, ids, rejectedImgPoints = aruco.detectMarkers(img_gray, aruco_dict, parameters=parameters)
font = cv.FONT_HERSHEY_SIMPLEX

rvec = None
tvec = None
if ids is not None:
    rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, aruco_size, mtx, dist)  # aruco码边长为0.028m
    (rvec - tvec).any()

    for i in range(rvec.shape[0]):
        aruco.drawAxis(img, mtx, dist, rvec[i, :, :], tvec[i, :, :], aruco_size)
        aruco.drawDetectedMarkers(img, corners, ids, (0, 179, 255))
    cv.putText(img, "Id: " + str(ids.T), (50, 80), font, 2, (0, 255, 0), 2, cv.LINE_AA)
    # cv.putText(warped, "{:.1f}".format(score), (10, 30),cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    # print(str(ids))
else:
    cv.putText(img, "No Ids", (50, 80), font, 2, (0, 0, 255), 3, cv.LINE_AA)
cv_show(img)


###计算位置
Camear_Point = None
for j in range(len(rvec)):
    dst, _ = cv.Rodrigues(rvec[j])
    dst = np.mat(dst)
    tr = tvec[j]
    tr = np.mat(tr)
    Camear_Point = -dst.I*tr.T
    Camear_Point = Camear_Point*1000    #单位转换为mm
    Xc = np.round(Camear_Point[0],2)
    Yc = np.round(Camear_Point[1],2)
    Zc = np.round(Camear_Point[2],2)
    print('\n',"-------------------------")
    print("Id: " + str(ids[j]) + '坐标为：')
    print('Xc=',Xc,'mm','Yc=',Yc,'mm','Zc=',Zc,'mm')

print(' ')




