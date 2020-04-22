import cv2 as cv
import cv2.aruco as aruco
import  numpy as np

def cv_show(img,name='Figure'):
    cv.namedWindow(name,cv.WINDOW_NORMAL)
    cv.imshow(name,img)
    cv.waitKey(0)
    cv.destroyAllWindows()



img1 = cv.imread(r'F:\PyCharm\Camera_calibration_GIT\aruco_identify_principle\003.png')
img1_gray = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)                   #灰度化
# cv_show(img1_gray)
img1_blur = cv.GaussianBlur(img1_gray,(3,3),1)
# cv_show(img1_blur)
edged = cv.Canny(img1_blur, 70, 200)
cv_show(edged)
cont,hierarchy = cv.findContours(edged,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)

#轮廓近似
approx = cv.approxPolyDP(cont[5],0.01*cv.arcLength(cont[5],True),True)
res1 = cv.drawContours(img1.copy(), approx, -1, (0, 0, 255), 20)
cv_show(res1)
for c in cont:

    res1 = cv.drawContours(img1.copy(),c,-1,(0,0,255),2)
    cv_show(res1)






