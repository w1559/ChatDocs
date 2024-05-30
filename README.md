
## 集成环境
    Python 3.12     
    开发工具 PyCharm
## Installation 安装
安装 [LangChain](https://github.com/hwchase17/langchain)和其他依赖的包。
```
>pip install -r requirements.txt
```
运行前需要使用hugging-face cli登录验证才可以使用其中的模型。运行下列脚本。根据提示输入自己的huggingface Access Token。Token获取位置[huggingface](https://huggingface.co/settings/tokens)
```
> huggingface-cli login
```
在config.py中填写你的API KEY
```
DASHSCOPE_API_KEY=""
```
运行
```
> streamlit run Chat.py
```
## 功能说明
使用在线OpenAI 的Embedding模型消耗的Token过高，决定使用HuggingFaceEmbeddings加载离线的Embedding模型。模型使用的是通义千问。向量数据库使用Chroma。

使用说明：上传文件，并对文件中的内容进行提问，系统会给出回答。