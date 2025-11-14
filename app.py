import os
import warnings
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
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
st.title("CTU CHATBOT")
st.markdown(
        """Xin chào! Tôi là trợ lý ảo của **Đại học Cần Thơ**.
        Tôi có thể giúp bạn về thông tin học vụ, đăng ký học phần, quy định sinh viên,
        hỗ trợ học bổng, học phí, và nhiều vấn đề khác. Hãy đặt câu hỏi nhé — tôi sẵn sàng hỗ trợ!""")

# LOAD DOCUMENTS & BUILD VECTORSTORE
directory = './data-rag/'
all_documents = []

if not os.path.exists(directory):
    st.error("Thư mục 'data-rag' chưa tồn tại.")
    st.stop()

# =====================================================================
# 1) LOAD TÀI LIỆU
@st.cache_resource
def load_documents():
    print("=== Load tài liệu lần đầu ===")
    directory = "./data-rag/"
    all_docs = []

    for root, dirs, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)

            if filename.endswith(".txt"):
                loader = TextLoader(filepath, encoding="utf-8")
            elif filename.endswith(".pdf"):
                loader = PyPDFLoader(filepath)
            elif filename.endswith(".docx"):
                loader = Docx2txtLoader(filepath)
            else:
                continue

            try:
                all_docs.extend(loader.load())
            except Exception as e:
                print("Lỗi load:", filepath, e)

    print("Tổng tài liệu:", len(all_docs))
    return all_docs


documents = load_documents()


# =====================================================================
# 2) CHUNKING
@st.cache_resource
def split_docs(_docs):
    print("=== Chia nhỏ văn bản lần đầu ===")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    texts = splitter.split_documents(_docs)
    print("Tổng chunk:", len(texts))
    return texts


texts = split_docs(documents)


# =====================================================================
# 3) LOAD EMBEDDINGS
@st.cache_resource
def load_embeddings():
    print("=== Load model embedding lần đầu ===")
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


embeddings = load_embeddings()


# =====================================================================
# 4) LOAD HOẶC TẠO CHROMA DB
@st.cache_resource
def load_vectorstore(_emb, _texts, _persist_dir="chroma_db"):

    db_exists = os.path.exists(_persist_dir) and len(os.listdir(_persist_dir)) > 0

    if not db_exists:
        print("=== Chưa có DB → Tạo mới ... ===")
        vs = Chroma.from_documents(
            _texts,
            _emb,
            collection_name="my_collection",
            persist_directory=_persist_dir
        )
        print("✓ DB đã tạo xong")
        return vs

    print("=== DB đã tồn tại → Load DB ===")
    vs = Chroma(
        collection_name="my_collection",
        embedding_function=_emb,
        persist_directory=_persist_dir
    )
    print("✓ DB đã load")
    return vs


vectorstore = load_vectorstore(embeddings, texts)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

llm = Ollama(model="llama3.2:3b", temperature=0.7)
system_prompt = """
Bạn là trợ lý AI thân thiện, trả lời bằng tiếng Việt, dễ hiểu, dành cho sinh viên Đại học Cần Thơ.

Hướng dẫn trả lời:
- Chỉ trả lời dựa trên tài liệu được cung cấp (context).  
- Nếu thông tin không có trong tài liệu, nói rõ rằng bạn **không tìm thấy thông tin phù hợp**.  
- Trình bày câu trả lời rõ ràng, ngắn gọn, có gạch đầu dòng hoặc số thứ tự nếu cần
- Không thêm dữ liệu hoặc suy đoán ngoài context.  
- Không lặp lại lịch sử hội thoại.  

Dữ liệu nền:
Lịch sử hội thoại:
{history}

Tài liệu liên quan:
{context}

"""


qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="Bạn là một trợ lý AI thân thiện, luôn trả lời hoàn toàn bằng tiếng Việt, diễn đạt tự nhiên và dễ hiểu."),
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
    # Hiển thị tin nhắn người dùng
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append(HumanMessage(content=prompt))

    print("=== BẮT ĐẦU TRUY VẤN ===")
    relevant_docs = retriever.invoke(prompt)
    print(f"Số tài liệu tìm thấy: {len(relevant_docs)}")

    if not relevant_docs:
        st.warning("Không tìm thấy tài liệu liên quan")
        context_documents_str = ""
    else:
        for i, doc in enumerate(relevant_docs):
            print(f"- Tài liệu {i+1}: {doc.metadata.get('source', 'Không có tên')} | {len(doc.page_content)} ký tự")
        context_documents_str = "\n\n".join(doc.page_content for doc in relevant_docs)

    # Lấy 3 lượt chat gần nhất để giữ ngữ cảnh
    history_text = ""
    for msg in st.session_state.messages[-3:]:
        role = "Người dùng" if isinstance(msg, HumanMessage) else "Trợ lý"
        history_text += f"{role}: {msg.content}\n"

    # Tạo prompt cho LLM
    qa_prompt_local = qa_prompt.partial(
        history=history_text,
        context=context_documents_str
    )

    chat_placeholder = st.chat_message("assistant")
    with chat_placeholder:
        message_placeholder = st.empty()
        message_placeholder.markdown("Đang trả lời...")

    # Gọi mô hình LLM
    llm_chain = {"input": RunnablePassthrough()} | qa_prompt_local | llm

    result = llm_chain.invoke(prompt)

    message_placeholder.markdown(result)

    # Lưu tin nhắn trả lời vào lịch sử
    st.session_state.messages.append(AIMessage(content=result))
