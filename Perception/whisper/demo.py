import whisper

# 要先安装ffmpeg: https://blog.csdn.net/qq_57171795/article/details/122842989
# https://zhuanlan.zhihu.com/p/662906303

video_path = "examples/test.mp3"

#--------------------------这个也行-----------------------------
# model = whisper.load_model("base")
# # video_path = "G:/Intelligent_Design_debug/Perception/whisper/examples/test.mp3"
# result = model.transcribe(video_path)
# print(result["text"])
#-------------------------------------------------------------

model = whisper.load_model("base")
# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio(video_path)
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio).to(model.device)

# detect the spoken language
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

# decode the audio
options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)

# print the recognized text
print(result.text)