#*********************************************************************#
#*                           By Huang Wenjun                         *#
#*********************************************************************#
import cv2 as cv
import cv2.aruco as aruco
import  numpy as np



def cv_show(frame,name='Figure'):
    cv.namedWindow(name,cv.WINDOW_AUTOSIZE)
    cv.imshow(name,frame)
    cv.waitKey(0)
    cv.destroyAllWindows()

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
        cv.putText(frame, "Id: " + str(ids.T), (50, 80), font, 2, (0, 255, 0), 2, cv.LINE_AA)

        markerCenter = []
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
            cv.putText(frame, "Id: " + str(ids[j]) + str([Xc,Yc,Zc]), location1, font, 1, (0, 255, 0), 2, cv.LINE_AA)


            # 方法二：采用aruco码的平corners的点计算
            uv_point = corners[j][0][2]
            # answer = uv_point - right1
            # Xw, Yw, Zw = left3.I * answer
            # Zc = 1000
            Xw = Zc*(uv_point[0]-mtx[0][2])/mtx[0][0]
            Yw = Zc*(uv_point[1]-mtx[1][2])/mtx[1][1]
            Zw = Zc
            location2 = tuple(corners[j][0][2])
            cv.putText(frame, "Id: "+ str([int(Xw), int(Yw), int(Zw)]), location2, font, 1, (255, 0, 0), 2, cv.LINE_AA)



            #仅存在两个aruco码时，进行距离计算
            if len(rvec) == 2:
                #方法一：采用aruco码的平移矩阵和旋转矩阵计算（仅含有两个标识）
                if j == 0:
                    point1 = np.mat([Xc ,Yc ,Zc])
                if j == 1:
                    point2 = np.mat([Xc ,Yc ,Zc])
                    distance = np.sqrt((point1-point2)*((point1-point2).T))
                    distance = int(distance)
                    cv.putText(frame, "Distance: "+str(distance), (50, 200), font, 1, (0, 0, 255), 2, cv.LINE_AA)


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
                    distance_m = distance_pix*Zc/mtx[0][0]
                    distance = int(distance_m)
                    cv.putText(frame, "Corners distance: " + str(distance), (50, 300), font, 1, (0, 0, 255), 2, cv.LINE_AA)



    else:
        cv.putText(frame, "No Ids", (50, 80), font, 2, (0, 0, 255), 3, cv.LINE_AA)

    cv.imshow("Capture_Paizhao", frame)
    k = cv.waitKey(1) & 0xFF
    if k == ord('s'):  # 按下s键，进入下面的保存图片操作
        cv.imwrite(r"F:\PyCharm\Camera_calibration_GIT\class5\0" + str(index) + ".jpg", frame_copy)
        print(cap.get(3))
        print(cap.get(4))
        print("save" + str(index) + ".jpg successfuly!")
        print("-------------------------")
        index += 1
    elif k == ord('q'):  # 按下q键，程序退出
        break
cap.release()
cv.destroyAllWindows()





