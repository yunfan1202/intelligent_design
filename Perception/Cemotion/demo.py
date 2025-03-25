#按文本字符串分析
# https://github.com/Cyberbolt/Cemotion
from cemotion import Cemotion

# 检查路径下是否存在模型，没有的话就下载模型
import os
import requests
import zipfile
from io import BytesIO

# 检查bert-base-chinese模型文件夹是否存在，没有就去下载并解压
if not os.path.exists("bert-base-chinese"):
    print("开始下载bert-base-chinese...")

    url = "https://github.com/yunfan1202/intelligent_design/releases/download/checkpoints/bert-base-chinese.zip"
    response = requests.get(url)
    # 直接解压到当前目录
    with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
        zip_ref.extractall()

    print("下载解压完成")
else:
    print("bert-base-chinese模型文件夹已存在")

# 检查cemotion_2.0.pt模型文件是否存在，没有就去下载并解压
if not os.path.exists("cemotion_2.0.pt"):
    print("开始下载cemotion_2.0.pt...")

    url = "https://github.com/yunfan1202/intelligent_design/releases/download/checkpoints/cemotion_2.0.pt"
    r = requests.get(url)
    with open("cemotion_2.0.pt", "wb") as f:
        f.write(r.content)
    print("下载完成！")
else:
    print("cemotion_2.0.pt模型文件已存在")


# 以下开始使用Cemotion进行情感分类
str_text1 = '配置顶级，不解释，手机需要的各个方面都很完美'
str_text2 = '院线看电影这么多年以来，这是我第一次看电影睡着了。简直是史上最大烂片！没有之一！侮辱智商！大家小心警惕！千万不要上当！再也不要看了！'

c = Cemotion()
print('"', str_text1 , '"\n' , '预测值:{:6f}'.format(c.predict(str_text1) ) , '\n')
print('"', str_text2 , '"\n' , '预测值:{:6f}'.format(c.predict(str_text2) ) , '\n')

list_text = [
    '内饰蛮年轻的，而且看上去质感都蛮好，貌似本田所有车都有点相似，满高档的！',
    '总而言之，是一家不会再去的店。',
    '湖大真好看，桃子湖真漂亮',
    '你好，初次见面，请多指教'
]

for each in c.predict(list_text):
    print(each)
