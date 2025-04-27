# Please install OpenAI SDK first: `pip3 install openai`

# 提示库：https://api-docs.deepseek.com/zh-cn/prompt-library/
# 参考API：https://api-docs.deepseek.com/zh-cn/
# Token计算器：https://console.cloud.tencent.com/hunyuan/tokenizer

from openai import OpenAI
import os

# 有VPN又不想关的话可以加上以下两句
os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"

# print(os.environ["DeepSeek_API_Key"])
client = OpenAI(api_key=os.environ["DeepSeek_API_Key"], base_url="https://api.deepseek.com")
# print("可用模型列表: ", client.models.list())  # 列出可用的模型列表


def demo():
    print("尝试deepseek-chat demo:")
    response = client.chat.completions.create(
        model="deepseek-chat",  # model='deepseek-chat' 即可调用 DeepSeek-V3；model='deepseek-reasoner'，即可调用DeepSeek-R1
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "你好，你是谁？9.11和9.8哪个更大?"},     # "你好，你可以做些什么智能功能？请举几个例子给我看看"
        ],
        stream=False
    )
    print(response.choices[0].message.content)


def demo_reasoner():
    print("尝试deepseek-reasoner demo:")
    # Round 1
    messages = [{"role": "user", "content": "9.11和9.8哪个更大?"}]
    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=messages,
    )

    reasoning_content = response.choices[0].message.reasoning_content
    content = response.choices[0].message.content

    print("reasoning_content:", reasoning_content)
    print("content:", content)
    print("-----------------------------------------------------------------------------------------------")
    # Round 2
    messages.append({"role": "assistant", "content": content})
    messages.append({'role': 'user', 'content': "'strawberry'这个单词里有几个r?"})
    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=messages,
    )

    print("response_content:", response.choices[0].message.content)


demo()

demo_reasoner()