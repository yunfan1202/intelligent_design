# https://github.com/openai/CLIP
# pip install ftfy regex tqdm
import torch
import clip
from PIL import Image

# 检查路径下是否存在模型，没有的话就下载模型
import os
import requests

folder = os.path.join(os.getcwd(), "checkpoints")
target_file = os.path.join(folder, "ViT-B-32.pt")
os.makedirs(folder, exist_ok=True)
if not os.path.isfile(target_file):
    print("开始下载模型文件...")
    url = "https://github.com/yunfan1202/intellegent_design/releases/download/checkpoints/ViT-B-32.pt"
    r = requests.get(url)
    with open(target_file, "wb") as f:
        f.write(r.content)
    print("下载完成！")
else:
    print("ViT-B-32.pt文件已存在")

# 以下开始使用CLIP进行零样本分类
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("checkpoints/ViT-B-32.pt", device=device)

image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    print(image_features.shape, text_features.shape)
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]