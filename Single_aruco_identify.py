import cv2 as cv
import cv2.aruco as aruco
import  numpy as np



with np.load(r'F:\PyCharm\Camera_calibration_GIT\class1\class_mtx.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]
print("#######加载相机内参和畸变矩阵#######")
print(mtx, '\n\n', dist)


index = 0
frame = cv.imread(r'F:\PyCharm\Camera_calibration_GIT\video collection\Aruco test\02.jpg')
img = frame.copy()
img_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
# cv_show(frame)
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters_create()

corners, ids, rejectedImgPoints = aruco.detectMarkers(img_gray, aruco_dict, parameters=parameters)
font = cv.FONT_HERSHEY_SIMPLEX

if ids is not None:
    rvec, tvec, objPoints = aruco.estimatePoseSingleMarkers(corners, 0.076, mtx, dist)  #aruco码边长为0.076m
    (rvec-tvec).any()
    R, _ = cv.Rodrigues(rvec)
    print("translation is  ", tvec*1000)

    for i in range(rvec.shape[0]):
        aruco.drawAxis(frame, mtx, dist, rvec[i, :, :], tvec[i, :, :], 0.076)
        aruco.drawDetectedMarkers(frame, corners)
    cv.putText(frame, "Id: " + str(ids.T), (50, 80), font, 2, (0, 255, 0), 2, cv.LINE_AA)
    # cv.putText(warped, "{:.1f}".format(score), (10, 30),cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    # print(str(ids))
else:
    cv.putText(frame, "No Ids", (50, 80), font, 2, (0, 0, 255), 3, cv.LINE_AA)


sita_x = dg(R[0][0])
sita_y = dg(R[1][0])
sita_z = dg(R[2][0])

print("sita_x is  ", sita_x,'度')
print("sita_y is  ", sita_y,'度')
print("sita_z is  ", sita_z,'度')

cv_show(frame)

# 00.jpg
# cv.c ircle(img,(913,509),2,(0,0,255),2)
# cv.circle(img,(931,509),2,(0,0,255),2)
# cv.circle(img,(931,530),2,(0,0,255),2)
# cv.circle(img,(913,5 30),2,(0,0,255),2)

# 02.jpg internal block
cv.circle(img,(380,393),2,(0,0,255),2)
cv.circle(img,(507,410),2,(0,0,255),2)
cv.circle(img,(370,458),2,(0,0,255),2)
cv.circle(img,(499,470),2,(0,0,255),2)
#02.jpg
cv.circle(img,(360,134),6,(0,255,0),4)
cv.circle(img,(702,220),6,(0,255,0),4)
cv.circle(img,(655,545),6,(0,255,0),4)
cv.circle(img,(287,516),6,(0,255,0),4)


for i in range(len(ids)):
    markerCenter = corners[i][0].sum(0) / 4.0
    # a = markerCenter.reshape([1, -1, 2])
    markerCenterIdeal = cv.undistortPoints(markerCenter.reshape([1, -1, 2]), mtx, dist)
    markerCameraCoodinate = np.append(markerCenterIdeal[0][0], [1])
    print('++++++++markerCameraCoodinate')
    print(markerCameraCoodinate)


cv.circle(img,(markerCenter[0],markerCenter[1]),3,(255,0,0),4)

cv_show(img)