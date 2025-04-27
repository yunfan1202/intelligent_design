import os
from openai import OpenAI
# 腾讯混元：https://console.cloud.tencent.com/hunyuan/start
# API文档：https://cloud.tencent.com/document/product/1729/111007
# 腾讯混元模型list：https://cloud.tencent.com/document/product/1729/104753
# 腾讯混元模型计费：https://cloud.tencent.com/document/product/1729/97731
# 阿里千问模型计费：https://help.aliyun.com/zh/model-studio/models?spm=a2c4g.11186623.0.0.4fd51e19GHFdUT

# 有VPN又不想关的话可以加上以下两句
os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"

# 构造 client
client = OpenAI(
    api_key=os.environ.get("HUNYUAN_API_KEY"),  # 混元 APIKey
    base_url="https://api.hunyuan.cloud.tencent.com/v1",  # 混元 endpoint
)


completion = client.chat.completions.create(
    model="hunyuan-lite",      # hunyuan-lite, hunyuan-turbos-latest
    messages=[
        {
            "role": "user",
            "content": "以后如果我说的话以'喵'结尾，那么你的回答也必须以'喵'结尾，明白了吗"
        },
        {
            "role": "assistant",
            "content": "明白了"
        },
        {
            "role": "user",
            "content": "你是谁喵？"
        }
    ],
    extra_body={
        "enable_enhancement": True,  # <- 自定义参数
    },
)

# completion = client.chat.completions.create(
#     model="hunyuan-lite",      # hunyuan-lite, hunyuan-turbos-latest
#     messages=[
#         {
#             "Role": "system",
#             "Content": "将英文单词转换为包括中文翻译、英文释义和一个例句的完整解释。请检查所有信息是否准确，并在回答时保持简洁，不需要任何其他反馈。"
#         },
#         {
#             "Role": "user",
#             "Content": "请翻译：湖南大学设计艺术学院牛逼！"
#         }
#     ],
#     extra_body={
#         "enable_enhancement": True,  # <- 自定义参数
#     },
# )

print(completion.choices[0].message.content)