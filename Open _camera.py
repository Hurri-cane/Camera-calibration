import cv2 as cv


cap = cv.VideoCapture(1,cv.CAP_DSHOW)  #更改API设置
flag = cap.isOpened()
cap.set(3, 1280)
cap.set(4, 720)


index = 0
while (flag):
    ret, frame = cap.read()

    # frame = cv.flip(frame, 1)  # 水平翻转
    cv.imshow("Capture_Paizhao", frame)
    # print(cap.get(3),cap.get(4))
    k = cv.waitKey(1) & 0xFF
    if k == ord('s'):  # 按下s键，进入下面的保存图片操作
        cv.imwrite(r"F:\PyCharm\Camera calibration\class1\0" + str(index) + ".jpg", frame)
        print(cap.get(3))
        print(cap.get(4))
        print("save" + str(index) + ".jpg successfuly!")
        print("-------------------------")
        index += 1
    elif k == ord('q'):  # 按下q键，程序退出
        break
cap.release()
cv.destroyAllWindows()
