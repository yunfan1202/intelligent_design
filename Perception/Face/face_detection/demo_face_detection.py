# keras的人脸情绪、性别分类 https://github.com/Furkan-Gulsen/face-classification
# 好像很齐全，但是是TensorFlow的 https://github.com/oarriaga/paz
# ------------------------------------------------------------------------------------------------
# --------------本代码参考https://blog.csdn.net/LOVEmy134611/article/details/121006385--------------
# ------------------------------------------------------------------------------------------------
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 可视化函数
def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]
    ax = plt.subplot(2, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=8)
    plt.axis('off')

def show_detection(image, faces):
    """在每个检测到的人脸上绘制一个矩形进行标示"""
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 5)
    return image


# 加载图像
img = cv2.imread("290035.jpg")
# 将 BGR 图像转换为灰度图像


def alg_opencv():
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 加载分类器
    cas_alt2 = cv2.CascadeClassifier("haarcascade_model/haarcascade_frontalface_alt2.xml")
    cas_default = cv2.CascadeClassifier("haarcascade_model/haarcascade_frontalface_default.xml")
    # 检测人脸
    faces_alt2 = cas_alt2.detectMultiScale(gray)
    faces_default = cas_default.detectMultiScale(gray)
    retval, faces_haar_alt2 = cv2.face.getFacesHAAR(img, "haarcascade_model/haarcascade_frontalface_alt2.xml")
    faces_haar_alt2 = np.squeeze(faces_haar_alt2)
    retval, faces_haar_default = cv2.face.getFacesHAAR(img, "haarcascade_model/haarcascade_frontalface_default.xml")
    faces_haar_default = np.squeeze(faces_haar_default)
    # 绘制人脸检测框
    img_faces_alt2 = show_detection(img.copy(), faces_alt2)
    img_faces_default = show_detection(img.copy(), faces_default)
    img_faces_haar_alt2 = show_detection(img.copy(), faces_haar_alt2)
    img_faces_haar_default = show_detection(img.copy(), faces_haar_default)
    # 可视化
    show_img_with_matplotlib(img_faces_alt2, "detectMultiScale(frontalface_alt2): " + str(len(faces_alt2)), 1)
    show_img_with_matplotlib(img_faces_default, "detectMultiScale(frontalface_default): " + str(len(faces_default)), 2)
    show_img_with_matplotlib(img_faces_haar_alt2, "getFacesHAAR(frontalface_alt2): " + str(len(faces_haar_alt2)), 3)
    show_img_with_matplotlib(img_faces_haar_default, "getFacesHAAR(frontalface_default): " + str(len(faces_haar_default)), 4)
    plt.show()

def alg_opencv_dnn():
    # 加载预训练的模型， Caffe 实现的版本
    net = cv2.dnn.readNetFromCaffe("dnn_model/deploy.prototxt", "dnn_model/res10_300x300_ssd_iter_140000_fp16.caffemodel")
    # 加载预训练的模型， Tensorflow 实现的版本
    # net = cv2.dnn.readNetFromTensorflow("opencv_face_detector_uint8.pb", "opencv_face_detector.pbtxt")
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104., 117., 123.], False, False)
    # 将 blob 设置为输入并获取检测结果
    net.setInput(blob)
    detections = net.forward()


    detected_faces = 0
    w, h = img.shape[1], img.shape[0]
    # 迭代所有检测结果
    for i in range(0, detections.shape[2]):
        # 获取当前检测结果的置信度
        confidence = detections[0, 0, i, 2]
        # 如果置信大于最小置信度，则将其可视化
        if confidence > 0.7:
            detected_faces += 1
            # 获取当前检测结果的坐标
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')
            # 绘制检测结果和置信度
            text = "{:.3f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(img, (startX, startY), (endX, endY), (255, 0, 0), 3)
            cv2.putText(img, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # 可视化
    show_img_with_matplotlib(img, "DNN face detector: " + str(detected_faces), 1)
    plt.show()


# alg_opencv()
alg_opencv_dnn()