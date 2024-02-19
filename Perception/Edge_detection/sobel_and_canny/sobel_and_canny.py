# https://blog.csdn.net/qq_36552489/article/details/105144535
import cv2


def sobel():
    img = cv2.imread('36046.jpg') # 读取图像
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # 转化为灰度图
    blur = cv2.GaussianBlur(img_gray, (3, 3), 0) # 高斯滤波处理原图像降噪

    x = cv2.Sobel(blur, cv2.CV_16S, 1, 0)   # Sobel函数求完导数后会有负值，还有会大于255的值
    y = cv2.Sobel(blur, cv2.CV_16S, 0, 1)   # 使用16位有符号的数据类型，即cv2.CV_16S
    Scale_absX = cv2.convertScaleAbs(x)  # 转回uint8
    Scale_absY = cv2.convertScaleAbs(y)
    sobel_image = cv2.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)
    # cv2.imshow("absX_process", Scale_absX)
    # cv2.imshow("absY_process", Scale_absY)
    # cv2.imshow('sobel_xy_process', sobel_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite('36046_sobel.jpg', sobel_image)


def canny():
    img = cv2.imread('36046.jpg') # 读取图像

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # 转化为灰度图
    blur = cv2.GaussianBlur(img_gray, (3, 3), 0)  # 高斯滤波处理原图像降噪

    canny_image = cv2.Canny(blur, 100, 200)   # 双阈值自己调
    cv2.imwrite('36046_canny.jpg', canny_image)


sobel()
canny()
