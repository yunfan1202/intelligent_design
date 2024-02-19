import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np

from src import model
from src import util
from src.body import Body
from src.hand import Hand

body_estimation = Body('model/body_pose_model.pth')
hand_estimation = Hand('model/hand_pose_model.pth')

def process_frame(frame, body=True, hands=True):
    canvas = copy.deepcopy(frame)
    if body:
        candidate, subset = body_estimation(frame)
        canvas = util.draw_bodypose(canvas, candidate, subset)
    if hands:
        hands_list = util.handDetect(candidate, subset, frame)
        all_hand_peaks = []
        for x, y, w, is_left in hands_list:
            peaks = hand_estimation(frame[y:y+w, x:x+w, :])
            peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
            peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
            all_hand_peaks.append(peaks)
        # print("canvas.shape:", canvas.shape) # (720, 1280, 3)
        canvas = util.draw_handpose(canvas, all_hand_peaks)
    return canvas


detect_hands = True
detect_body = True
video_file = "videos/video.avi"
cap = cv2.VideoCapture(video_file)

fps = cap.get(cv2.CAP_PROP_FPS)
# size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# print("fps:", fps, "size:", size) # fps: 50.0 size: (1280, 720)

# 定义编码格式mpge-4
# 一种视频格式，参数搭配固定，不同的编码格式对应不同的参数
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# 定义视频文件输入对象
output_file = ".".join(video_file.split(".")[:-1])+"_pose.mp4"

frame_num = 0
# 循环使用cv2的read()方法读取视频帧
while(cap.isOpened()):
    frame_num += 1
    ret, frame = cap.read()
    print("processing frame", frame_num)
    if frame is None:
        break
    # if frame_num == 10:
    #     break
    posed_frame = process_frame(frame, body=detect_body, hands=detect_hands)
    cv2.imshow('posed_frame', posed_frame)

    h, w = posed_frame.shape[:2]
    if frame_num == 1:
        outVideo = cv2.VideoWriter(output_file, fourcc, fps, (w, h))  # 第一个参数是保存视频文件的绝对路径
    outVideo.write(posed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# 释放窗口
cap.release()
outVideo.release()
cv2.destroyAllWindows()