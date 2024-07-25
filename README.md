# 第一天总结

```json
{
  "version": 4,
  "history": [
    {
      "id": "ade3bac5-d197-4ce0-898d-e4896eaf5f27",
      "name": "HomeWork1",
      "messages": [
        {
          "role": "user",
          "content": "课程的目录如下：\n上午：\n1.引言：深度解读AI 发展四轮浪潮\n2.AI 大模型技术四阶技术总览\n\t• 提示工程（Prompt Engineering）\n\t• AI智能体（Agents）\n\t• 大模型微调（Fine-tuning）\n\t• 预训练技术（Pre-training）\n3.生成式人工智能前沿研究\n\t• OpenAIGPTs 与Assistants API 介绍\n\t• Google Gemini 大模型介绍\n\t• AI大模型技术未来发展趋势与挑战\n4.AI大模型创投与应用案例研究\n\n下午：\n1.初探大模型:起源与发展\n\t• 预热篇:解码注意力机制(Attention) \n\t• 变革里程碑:Transformer的崛起\n\t• 走向不同:GPT与Bert的选择\n2.GPT 模型家族:从始至今\n\t• 从GPT-1到GPT-3.5: 一路的⻛云变幻\n\t• ChatGPT: 赢在哪里\n\t• GPT-4: 一个新的开始\n3.提示学习(Prompt Learning) \n\t• 思维链(Chain-of-Thought, CoT): 开山之作\n\t• 自洽性(Self-Consistency): 多路径推理\n\t• 思维树(Tree-of-Thoughts, ToT): 续写佳话\n4.基于GPT 的提示工程最佳实践\n5.Playground Chat: 实践 GPT 大模型提示工程"
        },
        {
          "role": "assistant",
          "content": "本节AI课程深入剖析了人工智能发展的四轮浪潮，揭示了AI大模型技术的核心组成部分，包括提示工程、AI智能体、模型微调和预训练技术。课程还探讨了生成式AI的最新研究进展，包括OpenAI的GPTs和Assistants API，Google的Gemini大模型，以及AI大模型技术面临的未来趋势和挑战。此外，还涉及了AI大模型在创投和应用案例中的实际运用。\n\n下午部分重点介绍了大模型的起源与发展，包括注意力机制、Transformer架构的重要性，以及GPT和Bert两种不同路径的演进。GPT模型家族的发展历程和ChatGPT的成功要素，以及GPT-4的新起点也被详细讲解。进一步地，课程深入了解了提示学习，包括思维链、自洽性和思维树等概念。最后，通过基于GPT的提示工程最佳实践和Playground Chat实践活动，使学员能够将理论知识应用于实际操作中。"
        }
      ],
      "model": {
        "id": "gpt-4-turbo-deploy",
        "name": "GPT-4"
      },
      "prompt": "你接下来将帮助制作一节AI课程的总结请求，请你为他分点分段写出使用书面语言构成的总结，不要啰嗦，不要照抄，不要使用首先，其次等词语",
      "temperature": 0.5,
      "folderId": null
    }
  ],
  "folders": [],
  "prompts": []
}
```

# 手写文字识别
[手写图片](./openai_api/images/athelas_test.png)

```python
from openai import OpenAI
from IPython.display import display, Markdown

import base64
import requests
import json

client = OpenAI()

def query_base64_image_description(image_path, prompt="解释下图里的内容？", max_tokens=1000):

    # 实现 Base64 编码
    def encode_image(path):
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    # 获取图像的 Base64 编码字符串
    base64_image = encode_image(image_path)

    # 构造请求的 HTTP Header
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {client.api_key}"
    }

    # 构造请求的负载
    payload = {
        "model": "gpt-4-turbo",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        "max_tokens": max_tokens
    }

    # 发送 HTTP 请求
    response = requests.post("https://api.xiaoai.plus/v1/chat/completions", headers=headers, json=payload)

    # 检查响应并提取所需的 content 字段
    if response.status_code == 200:
        response_data = response.json()
        content = response_data['choices'][0]['message']['content']
        return content
    else:
        return f"Error: {response.status_code}, {response.text}"

content = query_base64_image_description("./images/athelas_test.png",prompt="使用MarkDown解释下图里的内容？")
# 使用 display 和 Markdown 函数显示 Markdown 内容
display(Markdown(content))
```

图中内容可以用Markdown语法如下表示：

```markdown
Athelas Wang  
2024年 OpenAI加入
```

这段Markdown文本表示了两行文字：“Athelas Wang” 和 “2024年 OpenAI加入”，第一行为人名，第二行表示该人在2024年加入了OpenAI。

# 翻译器添加其他语言
```bash
python ai_translator/main.py --model_type OpenAIModel --file_format markdown --book tests/test.pdf --openai_model gpt-3.5-turbo --target_language 日文
```

> 2024-07-23 14:33:29.574 | INFO     | translator.writer:_save_translated_book_markdown:83 - pdf_file_path: tests/test.pdf
> 2024-07-23 14:33:29.574 | INFO     | translator.writer:_save_translated_book_markdown:84 - 开始翻译: tests/test_translated.md
> 2024-07-23 14:33:29.574 | INFO     | translator.writer:_save_translated_book_markdown:108 - 翻译完成: tests/test_translated.md

テストデータ
このデータセットには、OpenAIのAI言語モデルであるChatGPTによって提供された2つのテストサンプルが含まれています。
これらのサンプルには、テキストとテーブルの形式の両方をサポートする英語から中国語への翻訳ソフトウェアをテストするために使用できる、マークダウンテーブルと英語のテキストパッセージが含まれています。
テキストのテスト
The quick brown fox jumps over the lazy dog. This pangram contains every letter of the English
alphabet at least once. Pangrams are often used to test fonts, keyboards, and other text-related
tools. In addition to English, there are pangrams in many other languages. Some pangrams are more
difficult to construct due to the unique characteristics of the language.
テーブルのテスト

| [フルーツ、色、価格（USD）] |
| --- |
| [リンゴ、赤、1.20] |
| [バナナ、黄色、0.50] |
| [オレンジ、オレンジ、0.80] |
| [イチゴ、赤、2.50] |
| [ブルーベリー、青、3.00] |
| [キウイ、緑、1.00] |
| [マンゴー、オレンジ、1.50] |
| [ぶどう、紫、2.00] |

---

# 汉语言文学老师
```
> Entering new MultiPromptChain chain...

> Entering new LLMRouterChain chain...

> Finished chain.
None: {'input': '小黑子是什么鸡？'}
> Finished chain.
{'input': '小黑子是什么鸡？', 'history': '', 'text': ' 小黑子是一种原产于中国的鸡，也被称为“黑羽鸡”。它的身体呈黑色，羽毛有光泽，并且具有很高的肉质，因此被广泛用于食用。它们通常在山区生长，喜欢吃虫子和杂草，所以肉质更为健康和美味。它们也被认为是一种珍贵的养殖鸡类，因为它们的繁殖能力很强，且
```

```
> Entering new MultiPromptChain chain...


> Entering new LLMRouterChain chain...

> Finished chain.
计算机: {'input': '电脑是什么生物的脑？'}
> Finished chain.
{'input': '电脑是什么生物的脑？', 'text': '\n\n电脑并不是生物，它是一种由人类设计和制造的工具。它的运行原理是基于电子技术和程序编码，可以进行数据处理、存储和输出等功能。因此，它可以被看作是一种智能化的机器，但并没有生命。'}
```

```
> Entering new MultiPromptChain chain...

> Entering new LLMRouterChain chain...

> Finished chain.
汉语言: {'input': '汉语言是刻在石头还是龟壳上？'}
> Finished chain.
{'input': '汉语言是刻在石头还是龟壳上？', 'text': '\n\n这个问题涉及到两个方面：汉字的起源和古代的书写材料。首先，汉语言的起源可以追溯到古代的甲骨文，这是刻在龟壳或兽骨上的文字。随着时间的推移，汉字的书写材料也多样化，包括石头、竹简、纸张等。因此，汉语言既可以刻在石头上，也可以刻在龟壳上，取决于当时的书写需求和可用材料。'}
```

# 航运仓位销售
[结果展示](./langchain/container_sales_chatbot/screenshot.png)