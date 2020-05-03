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

with np.load(r'F:\PyCharm\Camera_calibration_GIT\class1\class_mtx.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]

print("#######加载相机内参和畸变矩阵#######")
print(mtx, '\n\n', dist)
index = 0



cap = cv.VideoCapture(r'F:\PyCharm\Camera_calibration_GIT\video collection\aruco_video _phone.mov')
flag = cap.isOpened()
cv.namedWindow('video',0)
cv.resizeWindow('video',1280,720)
# cap.set(3, 1280)
# cap.set(4, 720)
while (flag):
    ret, frame = cap.read()
    # frame = cv.resize(frame,(1280,720))
    # cv_show(frame)
    frame_copy = frame.copy()
# else:
#     frame = cv.imread(r'F:\PyCharm\Camera calibration\video collection\collection_2\03.jpg')

    img_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # cv_show(frame)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters_create()

    corners, ids, rejectedImgPoints = aruco.detectMarkers(img_gray, aruco_dict, parameters=parameters)
    font = cv.FONT_HERSHEY_SIMPLEX

    if ids is not None:
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.076, mtx, dist)  #aruco码边长为0.076m
        (rvec-tvec).any()

        for i in range(rvec.shape[0]):
            aruco.drawAxis(frame, mtx, dist, rvec[i, :, :], tvec[i, :, :], 0.076)
            aruco.drawDetectedMarkers(frame, corners,ids,(0, 179, 255))

        cv.putText(frame, "Id: " + str(ids.T), (50, 80), font, 2, (0, 255, 0), 2, cv.LINE_AA)
        # cv.putText(warped, "{:.1f}".format(score), (10, 30),cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        # print(str(ids))
    else:
        cv.putText(frame, "No Ids", (50, 80), font, 2, (0, 0, 255), 3, cv.LINE_AA)
    cv.imshow("video", frame)
    k = cv.waitKey(1) & 0xFF
    if k == ord('s'):  # 按下s键，进入下面的保存图片操作
        cv.imwrite(r"F:\PyCharm\Camera_calibration_GIT\class4\0" + str(index) + ".jpg", frame_copy)
        print(cap.get(3))
        print(cap.get(4))
        print("save" + str(index) + ".jpg successfuly!")
        print("-------------------------")
        index += 1
    elif k == ord('q'):  # 按下q键，程序退出
        break
cap.release()
cv.destroyAllWindows()




