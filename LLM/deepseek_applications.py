# Please install OpenAI SDK first: `pip3 install openai`

# 提示库：https://api-docs.deepseek.com/zh-cn/prompt-library/
# 参考API：https://api-docs.deepseek.com/zh-cn/

from openai import OpenAI
import os

# 有VPN又不想关的话可以加上以下两句
os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"

# print(os.environ["DeepSeek_API_Key"])
client = OpenAI(api_key=os.environ["DeepSeek_API_Key"], base_url="https://api.deepseek.com")
# print("可用模型列表: ", client.models.list())  # 列出可用的模型列表


def demo(prompts):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": prompts["sys_prompt"]},
            {"role": "user", "content": prompts["user_content"]},
        ],
        stream=False
    )

    return response.choices[0].message.content


prompt_translate = {
    "sys_prompt": "你是一名专业翻译，根据上下文语境进行中英互译",
    # "sys_prompt": "你是一名专业翻译，根据上下文语境进行中英互译(仅输出翻译的文字，不需要任何额外解释)",
    "user_content": "这个project的schedule有些问题，cost偏高。目前我们没法confirm手上的resource能完全take得了"
}

prompt_polish = {
    "sys_prompt": "请将以下文字改写成猫娘风格",  # 学术风格，小红书风格，猫娘风格
    "user_content": "我们发现这个算法跑得比之前的快好多，准确率也高了。"
}

prompt_emotion1 = {
    "sys_prompt": "分析用户输入文本的情绪，用单一标签输出：positive/neutral/negative",
    "user_content": "刚考完期中考试太爽了，今晚必须爽吃一波"
}

prompt_emotion2 = {
    "sys_prompt": "从以下六类选择最匹配的情绪：喜悦、悲伤、愤怒、惊讶、恐惧、厌恶",
    "user_content": "看到有人浪费美食，我气得浑身发抖"
}       # 愤怒

prompt_emotion3 = {
    "sys_prompt": "判断情绪类型(喜悦/焦虑/平静)并给出强度评分(1-5)，格式：类型:评分",  # 强度分级情感分析
    "user_content": "明天要面试了，既期待新机会又担心发挥不好"
}       # 焦虑：4

prompt_emotion4 = {
    "sys_prompt": "从[自豪,困惑,期待,失望]中选择所有适用标签，用逗号分隔",      # 多标签情感识别
    "user_content": "虽然实验失败了，但发现了新的研究方向"
}       # 失望、期待

prompt_emotion5 = {
    "sys_prompt": "识别用户对产品的情绪：质量抱怨/价格敏感/物流不满/功能赞赏",     # 领域特定情感分析（电商）
    "user_content": "手机拍照效果惊艳，但续航比宣传的差远了"
}       # 功能赞赏 + 质量抱怨


result = demo(prompt_polish)
print(result)


# def demo(prompt_pairs):
#     response = client.chat.completions.create(
#         model="deepseek-chat",
#         messages=[
#             {"role": "system", "content": "分析用户输入文本的情绪，用单一标签输出：positive/neutral/negative"},
#             {"role": "user", "content": "刚考完期中考试太爽了，今晚必须爽吃一波"},
#         ],
#         stream=False
#     )
#     print(response.choices[0].message.content)

# def demo_translate():
#     response = client.chat.completions.create(
#         model="deepseek-chat",
#         messages=[
#             {"role": "system", "content": "你是一名专业翻译，根据上下文语境进行中英互译"},
#             # {"role": "system", "content": "你是一名专业翻译，根据上下文语境进行中英互译，只给出翻译的文字即可，不需要输出任何额外解释"},
#             {"role": "user", "content": "这个project的schedule有些问题，cost偏高。目前我们没法confirm手上的resource能完全take得了"},
#         ],
#         stream=False
#     )
#     print(response.choices[0].message.content)

