from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
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


audio_file = "test.mp3"
# #-----------------------------modelscope------------------------------
# inference_pipeline = pipeline(
#     task=Tasks.emotion_recognition,
#     model="checkpoints/emotion2vec_base_finetuned", model_revision="v2.0.4")
# rec_result = inference_pipeline(audio_file, output_dir="./outputs", granularity="utterance", extract_embedding=False)
# print(rec_result[0])

#-----------------------------funasr------------------------------
from funasr import AutoModel
model = AutoModel(model="checkpoints/emotion2vec_base_finetuned", model_revision="v2.0.4")
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

