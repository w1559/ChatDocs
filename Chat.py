import os
import shutil
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import UnstructuredFileLoader,CSVLoader,PyMuPDFLoader
from langchain_community.llms import Tongyi
from langchain.chains.question_answering import load_qa_chain
import streamlit as st

from langchain.prompts import PromptTemplate
from config import DASHSCOPE_API_KEY
os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY

# 加载文档并分割成小段文本texts (chunks)
def load_and_split(path: str):
    # 导入文本
    _, file_extension = os.path.splitext(path)
    global loader
    if file_extension == ".txt":
        # 导入文本
        loader = UnstructuredFileLoader(path)
    elif file_extension == ".csv":
        # 导入CSV
        loader = CSVLoader(path)
    elif file_extension == ".pdf":
        # 导入CSV
        loader = PyMuPDFLoader(path)
    # 将文本转成 Document 对象
    data = loader.load()
    # print(f'documents:{len(data)}')

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    split_docs = text_splitter.split_documents(data)
    # print("split_docs size:", len(split_docs))
    # print(split_docs)
    return split_docs

#embeddings模型
model_name = 'maidalun1020/bce-embedding-base_v1'
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}

embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

def into_db(path: str):
    split_docs = load_and_split(path)
    persist_directory = './chroma/news_test'

    if(os.path.exists(persist_directory)):
        # 从已经持久化的数据库中加载 db
        db = Chroma(persist_directory=persist_directory,embedding_function=embeddings)
        db.add_documents(split_docs)
    else:
        # 创建新的数据库
        db = Chroma.from_documents(split_docs, embeddings, persist_directory=persist_directory)
        db.persist()

def reload_db():
    # 文件夹路径
    data_folder = 'data/'
    chroma_folder = 'chroma/'

    #   启动时删除已有文件 删除data文件夹下的所有文件
    for filename in os.listdir(data_folder):
        file_path = os.path.join(data_folder, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('发生错误:', e)

    # 删除chroma文件夹下的所有文件
    for filename in os.listdir(chroma_folder):
        file_path = os.path.join(chroma_folder, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('发生错误:', e)

reload_db()
# 用streamlit生成web界面
st.title('简单自建知识库问答系统') # 设置标题


uploaded_file = st.file_uploader('上传你的文件', type=['txt', 'pdf', 'csv'])
if uploaded_file is not None:
    file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}

    st.write(file_details)

    # 读取文件内容
    file_content = uploaded_file.getvalue()

    # 创建文件目录
    file_dir = './data/'

    if not os.path.exists(file_dir):
        os.mkdir(file_dir)

    # 定义文件路径
    filepath = os.path.join(file_dir, uploaded_file.name)

    # 写文件到本地
    with open(filepath, 'wb') as f:
        f.write(file_content)
    into_db(filepath)
    st.success("文件已经被成功保存在: " + filepath)
else:
    st.write("Please upload a file.")


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 创建一个容器来存储聊天记录
chat_container = st.container()


with st.form(key='my_form'):

    user_input = st.text_input(label='你:')
    submit_button = st.form_submit_button(label='发送')

    # 清空按钮可以放在你认为合适的地方，这里我选择放在下方
    clear_button = st.form_submit_button(label='清空')

if clear_button:
    st.session_state.chat_history = []

# 根据用户输入，生成回复
if submit_button:
    print(f"用户输入：{user_input}")
    # 根据用户输入，从向量数据库搜索得到相似度最高的texts
    db = Chroma(persist_directory="./chroma/news_test", embedding_function=embeddings)
    # 搜索得到与用户输入相似度最大、而彼此之间有差异的texts
    # k是返回的texts数量，默认为4
    similarDocs = db.similarity_search(user_input, k=4)


    # 定义PromptTemplate实例
    template = """你是一个问答机器人，你能对于用户上传的文档进行解答。当你被问及你是谁时，
    你应该这样回答：“我是智能问答机器人，我能帮助你回答关于文档内容的问题”。当你不知道问题的答案时请回答“你的文档中没有相关问题答案”。回答用户的问题，请基于以下信息：
    {context}

    问题: {question}
    回答:"""

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    llm=Tongyi()

    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    answer = chain.invoke(input={"input_documents": similarDocs, "question": user_input})

    # st.write("回答:")
    # st.text_area("回答内容", value=answer['output_text'], height=200, disabled=True)

    st.session_state.chat_history.append(("你", user_input))
    st.session_state.chat_history.append(("ChatGPT", answer['output_text']))

    chat_container.empty()
    # 清空用户输入
    st.session_state.user_input = ""
    with chat_container:
        for speaker, message in st.session_state.chat_history:
            if speaker == "你":
                with st.chat_message("you"):
                     st.markdown(message)
            else:
                with st.chat_message("assistant"):
                    st.markdown(message)
