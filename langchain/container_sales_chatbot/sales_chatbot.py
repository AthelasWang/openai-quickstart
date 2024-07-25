import gradio as gr

from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

def initialize_sales_bot():
    # 实例化文档加载器
    loader = TextLoader("./real_estate_sales_data.txt")
    # 加载文档
    documents = loader.load()
    # 实例化文本分割器
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    # 分割文本
    docs = text_splitter.split_documents(documents)
    # OpenAI Embedding 模型
    embeddings = OpenAIEmbeddings()
    # FAISS 向量数据库，使用 docs 的向量作为初始化存储
    db = FAISS.from_documents(docs, embeddings)
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.5)
    
    global SALES_BOT    
    SALES_BOT = RetrievalQA.from_chain_type(llm,
                                           retriever=db.as_retriever(search_type="similarity_score_threshold",
                                                                     search_kwargs={"score_threshold": 0.8}))
    SALES_BOT.combine_documents_chain.verbose = True
    # 返回向量数据库的检索结果
    SALES_BOT.return_source_documents = True

    return SALES_BOT

def sales_chat(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")
    # TODO: 从命令行参数中获取
    enable_chat = True

    ans = SALES_BOT({"query": message})
    # 如果检索出结果，或者开了大模型聊天模式
    # 返回 RetrievalQA combine_documents_chain 整合的结果
    if ans["source_documents"] or enable_chat:
        print(f"[result]{ans['result']}")
        print(f"[source_documents]{ans['source_documents']}")
        if len(ans['source_documents']) == 0:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "你是一个中国顶级的航运仓位销售。你需要回复用户关于选购航运仓位上的一些问题。不要回答除了航运仓位以外的问题，不要暴露你是个机器人"),
                ("user", "问题：{user_input}")
            ])
            chat = prompt.format_messages(user_input=message)
            llm = ChatOpenAI(model_name="gpt-4", temperature=0)
            chat_result = llm.invoke(chat)
            print(f"[chat_result]{chat_result.content}")
            return chat_result.content
        return ans["result"]
    # 否则输出套路话术
    else:
        return "这个问题我要问问领导"
    

def launch_gradio():
    demo = gr.ChatInterface(
        fn=sales_chat,
        title="航运仓位销售",
        # retry_btn=None,
        # undo_btn=None,
        chatbot=gr.Chatbot(height=600),
    )

    demo.launch(share=True, server_name="0.0.0.0")

if __name__ == "__main__":
    # 初始化房产销售机器人
    initialize_sales_bot()
    # 启动 Gradio 服务
    launch_gradio()
