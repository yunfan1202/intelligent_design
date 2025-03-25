# from modelscope.pipelines import pipeline
# from modelscope.utils.constant import Tasks
import numpy as np
'''
Using the emotion representation model
rec_result only contains {'feats'}
	granularity="utterance": {'feats': [*768]}
	granularity="frame": {feats: [T*768]}
'''

'''
Using the finetuned emotion recognization model
rec_result contains {'feats', 'labels', 'scores'}
	extract_embedding=False: 9-class emotions with scores
	extract_embedding=True: 9-class emotions with scores, along with features

9-class emotions:
    0: angry
    1: disgusted
    2: fearful
    3: happy
    4: neutral
    5: other
    6: sad
    7: surprised
    8: unknown
'''

# 检查路径下是否存在模型，没有的话就下载模型
import os
import requests
import zipfile
from io import BytesIO

target_dir = "checkpoints/emotion2vec_base_finetuned"
os.makedirs("checkpoints", exist_ok=True) # 创建checkpoints目录（如果不存在）
if not os.path.exists(target_dir):
    print("开始下载emotion2vec_base_finetuned...")
    url = "https://github.com/yunfan1202/intelligent_design/releases/download/checkpoints/emotion2vec_base_finetuned.zip"
    response = requests.get(url)
    with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
        zip_ref.extractall("checkpoints")  # 修改这里指定解压目录
    print("下载解压完成")
else:
    print("emotion2vec_base_finetuned模型文件夹已存在")


# --------------------------------------------------------------------------
audio_file = "test.mp3"
# #-----------------------------modelscope------------------------------
# inference_pipeline = pipeline(
#     task=Tasks.emotion_recognition,
#     model="checkpoints/emotion2vec_base_finetuned", model_revision="v2.0.4")
# rec_result = inference_pipeline(audio_file, output_dir="./outputs", granularity="utterance", extract_embedding=False)
# print(rec_result[0])

#-----------------------------funasr------------------------------
from funasr import AutoModel
model = AutoModel(model="emotion2vec_base_finetuned", model_revision="v2.0.4")
# audio_file = f"{model.model_path}/example/test.wav"
rec_result = model.generate(audio_file, output_dir="./outputs", granularity="utterance", extract_embedding=False)
print(rec_result)

emotions = rec_result[0]['labels']
scores = rec_result[0]['scores']

Z = zip(scores, emotions)
Z = sorted(Z, reverse=True)
scores, emotions = zip(*Z)
for i, j in zip(scores, emotions):
    print(j, '\t', i)

