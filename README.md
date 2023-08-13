# GuwenLLAMA-古汉语大模型

**GuwenLLAMA-7B**，此版本为demo版，基于Chinese-Alpaca-Pro-7B(https://github.com/ymcui/Chinese-LLaMA-Alpaca) 训练而来，在古汉语现代汉语翻译，古汉语断句方面有着不错的效果。

## 简介 Brief Introduction

**GuwenLLAMA** 是一个创新性的大语言模型项目，着眼于在古汉语领域拓展大语言模型的应用。当前，各大科技公司和高校正投入大量资源开发通用的大语言模型，OpenAI在通用领域的统治地位已经不可撼动。然而，特定领域（Specific-Domain）的大语言模型也是不可忽视的发展趋势。医疗、教育、金融、法律等领域都已经拥有了各自的专业模型，但遗憾的是，古汉语领域迄今尚未有相关模型问世。

为了促进大语言模型在古汉语领域的应用，我们开源了GuwenLLAMA项目，其中核心模块为古汉语大模型**GuwenLLAMA**。GuwenLLAMA专注于古文领域，该模型目前拥有古汉语翻译，古汉语断句的能力，可以为汉语学者，古文爱好者和教育工作者提供有力的支持。

引入GuwenLLAMA这样的古汉语语言模型，我们有望在古文领域取得重大突破。它将成为学者的得力助手，为研究和传承古汉语文化提供精确、高效的工具，助力汉语文化的传承和发展。

如果您有兴趣直接获取GuwenLLAMA模型，请访问HuggingFace以获取权重文件。我们期待古文LLAMA在古汉语领域的广泛应用，为汉语文化的研究和传承带来新的可能。

## Huggingface链接
上传中


## 未来计划 Todo

1. 增加多轮对话能力。
2. 增加古汉语命名实体识别等更多自然语言理解能力。
3. 尝试接入Langchain，允许LLM模型与外界数据源进行连接。
4. 增大数据量与模型大小，以期望获得更好的古汉语理解能力。

## 数据集 Dataset
本项目用于训练的数据集在Releases中，将他们下载后放在Data文件夹下<br>
本项目的原始数据集来自[Classical-Modern](https://github.com/NiuTrans/Classical-Modern)，为了增加模型理解能力做了以下处理：

1. 将原始数据集分为3个部分，分别是<br>
      a. 正常古汉语与现代汉语相互转换的数据<br>
      b. 将数据集中来自同一章节的句子合并成一个长句子再做古汉语和现代汉语互相转换<br>
      c. 将一部分输入数据改为无标点古汉语，输出位有标点古汉语<br>

2. 使用self-instruct随机生成一定数量的instruct，在结合原始数据作为input和output来生成指令数据集

3. 训练过程中，随机抽取50%的数据去除标点

## 效果展示

见image_show文件夹

## 推理

```bash
python inference.py  --base_model ../SaveModel  --interactive  --load_in_8bit  --instruction_and_input
