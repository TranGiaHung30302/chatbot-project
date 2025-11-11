import os
import warnings
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.llms import Ollama

# Ẩn cảnh báo
def warn(*args, **kwargs):
    pass
warnings.warn = warn
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Chatbot tài liệu", layout="centered")
st.title("Chatbot (RAG + Ollama)")

# LOAD DOCUMENTS & BUILD VECTORSTORE
directory = './data-rag/'
all_documents = []
if not os.path.exists(directory):
    st.error("Thư mục ' data-rag' chưa tồn tại.")
    st.stop()
for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)
    if filename.endswith('.txt'):
        loader = TextLoader(filepath, encoding='utf-8')
    elif filename.endswith('.pdf'):
        loader = PyPDFLoader(filepath)
    else:
        continue
    documents = loader.load()
    all_documents.extend(documents)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(all_documents)
persist_dir = "chroma_db"
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
if not os.path.exists(persist_dir) or len(os.listdir(persist_dir)) == 0:
    vectorstore = Chroma.from_documents(texts, embeddings, persist_directory=persist_dir)
else:
    vectorstore = Chroma(embedding_function=embeddings, persist_directory=persist_dir)

llm = Ollama(model="llama3.2:3b", temperature=0.7)
system_prompt = """
Bạn là một trợ lý AI thân thiện, luôn trả lời bằng tiếng Việt, tự nhiên và dễ hiểu.
Sử dụng thông tin trong tài liệu nếu có, nếu không hãy nói bạn không tìm thấy thông tin phù hợp.
Lịch sử hội thoại:
{history}
Tài liệu liên quan:
{context}
"""
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="Bạn là một trợ lý AI thân thiện, luôn trả lời hoàn toàn bằng tiếng Việt, diễn đạt tự nhiên và dễ hiểu.")
    ]
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

prompt = st.chat_input("Nhập câu hỏi...")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append(HumanMessage(content=prompt))
    # Tìm tài liệu liên quan
    relevant_docs = retriever.invoke(prompt)
    context_documents_str = "\n\n".join(doc.page_content for doc in relevant_docs)
    if not relevant_docs:
        st.warning("Không tìm thấy tài liệu liên quan, có thể context trống hoặc embedding lỗi.")
    # Gộp vào prompt
    qa_prompt_local = qa_prompt.partial(
        history=st.session_state.messages,
        context=context_documents_str
    )
    # Tạo chain
    llm_chain = {"input": RunnablePassthrough()} | qa_prompt_local | llm
    # Gọi mô hình
    result = llm_chain.invoke(prompt)
    # Hiển thị & lưu lịch sử
    with st.chat_message("assistant"):
        st.markdown(result)
    st.session_state.messages.append(AIMessage(content=result))