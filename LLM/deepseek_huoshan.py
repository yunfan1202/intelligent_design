import os
from openai import OpenAI
# 火山引擎：https://console.volcengine.com/ark/


# 有VPN又不想关的话可以加上以下两句
os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"

client = OpenAI(api_key=os.environ["Huoshan_API_Key"], base_url = "https://ark.cn-beijing.volces.com/api/v3")
# client = OpenAI(api_key = os.environ.get("ARK_API_KEY"), base_url = "https://ark.cn-beijing.volces.com/api/v3")

# Non-streaming:
print("----- standard request -----")
completion = client.chat.completions.create(
    model = "deepseek-r1-250120",  # your model endpoint ID
    messages = [
        {"role": "system", "content": "你是人工智能助手"},
        {"role": "user", "content": "设计学有哪些必学科目？"},
    ],
)
print(completion.choices[0].message.content)

# Streaming:
print("----- streaming request -----")
stream = client.chat.completions.create(
    model = "deepseek-r1-250120",  # your model endpoint ID
    messages = [
        {"role": "system", "content": "你是人工智能助手"},
        {"role": "user", "content": "常见的十字花科植物有哪些？"},
    ],
    stream=True
)

for chunk in stream:
    if not chunk.choices:
        continue
    print(chunk.choices[0].delta.content, end="")
print()