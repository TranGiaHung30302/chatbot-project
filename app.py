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

st.set_page_config(
    page_title="CTU AI Assistant", 
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS để tạo giao diện đẹp hơn
st.markdown("""
    <style>
    /* Background gradient */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Container chính */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Header */
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem !important;
        font-weight: 800 !important;
        text-align: center;
        margin-bottom: 0.5rem !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #4a5568;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(to right, #f7fafc, #edf2f7, #f7fafc);
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    
    /* Chat messages */
    .stChatMessage {
        background: white;
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        transition: transform 0.2s;
    }
    
    .stChatMessage:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
    }
    
    /* User message */
    [data-testid="stChatMessageContent"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        padding: 1rem;
    }
    
    /* Input box */
    .stChatInput {
        border-radius: 25px;
        border: 2px solid #667eea;
        padding: 0.75rem;
        font-size: 1rem;
    }
    
    .stChatInput:focus {
        border-color: #764ba2;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f7fafc 0%, #edf2f7 100%);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Typing indicator */
    @keyframes typing {
        0%, 100% { opacity: 0.3; }
        50% { opacity: 1; }
    }
    
    .typing-indicator {
        display: inline-flex;
        gap: 4px;
    }
    
    .typing-indicator span {
        width: 8px;
        height: 8px;
        background: #667eea;
        border-radius: 50%;
        animation: typing 1.4s infinite;
    }
    
    .typing-indicator span:nth-child(2) {
        animation-delay: 0.2s;
    }
    
    .typing-indicator span:nth-child(3) {
        animation-delay: 0.4s;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    /* Stats */
    .stat-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stat-label {
        color: #718096;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header với icon
st.markdown("<h1>🤖 CTU AI Assistant</h1>", unsafe_allow_html=True)
st.markdown("""
    <div class="subtitle">
        <strong>Xin chào! Tôi là trợ lý AI thông minh của Đại học Cần Thơ</strong><br>
        💡 Học vụ | 📚 Đăng ký học phần | 📋 Quy định sinh viên | 💰 Học bổng & Học phí<br>
        <em>Hãy đặt câu hỏi - Tôi luôn sẵn sàng hỗ trợ bạn 24/7!</em>
    </div>
""", unsafe_allow_html=True)

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

# Sidebar với thông tin (di chuyển xuống sau khi load xong documents và texts)
with st.sidebar:
    st.markdown("### 🎯 Thông tin hệ thống")
    st.markdown("---")
    
    # Stats
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{len(documents)}</div>
                <div class="stat-label">Tài liệu</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{len(texts)}</div>
                <div class="stat-label">Đoạn văn</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### ⚙️ Cài đặt")
    
    # Temperature slider
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.7
    
    temperature = st.slider(
        "🌡️ Độ sáng tạo",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.temperature,
        step=0.1,
        help="Giá trị cao hơn = câu trả lời sáng tạo hơn"
    )
    st.session_state.temperature = temperature
    
    # Top K slider
    if "top_k" not in st.session_state:
        st.session_state.top_k = 10
    
    top_k = st.slider(
        "📊 Số tài liệu tham khảo",
        min_value=3,
        max_value=15,
        value=st.session_state.top_k,
        step=1,
        help="Số lượng tài liệu liên quan để tìm kiếm"
    )
    st.session_state.top_k = top_k
    
    st.markdown("---")
    st.markdown("### 💬 Quản lý")
    
    if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
        st.session_state.messages = [
            SystemMessage(content="Bạn là một trợ lý AI thân thiện, luôn trả lời hoàn toàn bằng tiếng Việt, diễn đạt tự nhiên và dễ hiểu."),
        ]
        st.rerun()
    
    st.markdown("---")
    st.markdown("""
        <div class="info-box">
            <strong>🔒 Bảo mật</strong><br>
            Dữ liệu của bạn được<br>xử lý cục bộ và an toàn
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div style='text-align: center; color: #718096; font-size: 0.8rem; margin-top: 2rem;'>
            <strong>CTU AI Assistant v1.0</strong><br>
            Powered by Llama 3.2 & Chroma DB
        </div>
    """, unsafe_allow_html=True)

# Sử dụng top_k từ sidebar
if "top_k" not in st.session_state:
    st.session_state.top_k = 10

retriever = vectorstore.as_retriever(search_kwargs={"k": st.session_state.top_k})

# Sử dụng temperature từ sidebar
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7

llm = Ollama(model="llama3.2:3b", temperature=st.session_state.temperature)
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
        with st.chat_message("user", avatar="👤"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(message.content)

prompt = st.chat_input("💭 Nhập câu hỏi của bạn...")
if prompt:
    # Hiển thị tin nhắn người dùng với icon
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    st.session_state.messages.append(HumanMessage(content=prompt))

    print("=== BẮT ĐẦU TRUY VẤN ===")
    relevant_docs = retriever.invoke(prompt)
    print(f"Số tài liệu tìm thấy: {len(relevant_docs)}")

    if not relevant_docs:
        st.warning("⚠️ Không tìm thấy tài liệu liên quan")
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

    chat_placeholder = st.chat_message("assistant", avatar="🤖")
    with chat_placeholder:
        message_placeholder = st.empty()
        # Typing indicator với animation
        message_placeholder.markdown("""
            <div style='padding: 1rem;'>
                <div class='typing-indicator'>
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
                <span style='margin-left: 1rem; color: #667eea;'>Đang suy nghĩ...</span>
            </div>
        """, unsafe_allow_html=True)

    # Gọi mô hình LLM
    llm_chain = {"input": RunnablePassthrough()} | qa_prompt_local | llm

    result = llm_chain.invoke(prompt)

    # Hiển thị kết quả với format đẹp
    message_placeholder.markdown(f"""
        <div style='line-height: 1.8;'>
            {result}
        </div>
    """, unsafe_allow_html=True)

    # Lưu tin nhắn trả lời vào lịch sử
    st.session_state.messages.append(AIMessage(content=result))
