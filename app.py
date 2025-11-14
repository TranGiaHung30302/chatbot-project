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
st.title("Chatbot (RAG + Ollama)")

# LOAD DOCUMENTS & BUILD VECTORSTORE
directory = './data-rag/'
all_documents = []

if not os.path.exists(directory):
    st.error("Thư mục 'data-rag' chưa tồn tại.")
    st.stop()

print("=== BẮT ĐẦU LOAD TÀI LIỆU ===")
for root, dirs, files in os.walk(directory):
    for filename in files:
        filepath = os.path.join(root, filename)

        if filename.endswith('.txt'):
            loader = TextLoader(filepath, encoding='utf-8')
        elif filename.endswith('.pdf'):
            loader = PyPDFLoader(filepath)
        elif filename.endswith('.docx'):
            loader = Docx2txtLoader(filepath)
        else:
            print(f"Bỏ qua file không hỗ trợ: {filepath}")
            continue

        try:
            documents = loader.load()
            all_documents.extend(documents)
            # print(f"Đã load: {filepath} ({len(documents)} đoạn)")
        except Exception as e:
            print(f"Lỗi khi load {filepath}: {e}")

print(f"Tổng số tài liệu đã load: {len(all_documents)}")
if len(all_documents) == 0:
    print("CẢNH BÁO: Không có tài liệu nào được load. Kiểm tra lại thư mục data-rag.")
else:
    print("Đang chia nhỏ văn bản...")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)  # ưu tiên xuống dòng hơn khoảng trắng, tránh cắt giữa câu
texts = text_splitter.split_documents(all_documents)
print(f"Tổng số đoạn sau khi chia nhỏ: {len(texts)}")

persist_dir = "chroma_db"

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Sử dụng thiết bị cho embedding: {device}")

# embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# Kiểm tra DB hiện có
if not os.path.exists(persist_dir) or len(os.listdir(persist_dir)) == 0:
    print("Không tìm thấy DB cũ. Đang tạo mới...")
    vectorstore = Chroma.from_documents(
        texts,
        embeddings,
        collection_name="my_collection",
        persist_directory=persist_dir
    )
    # Không cần gọi persist() nữa
    print("Đã tạo DB thành công.")
else:
    print("Đang load DB có sẵn...")
    vectorstore = Chroma(
        collection_name="my_collection",
        embedding_function=embeddings,
        persist_directory=persist_dir
    )
    count = len(vectorstore.get()['ids'])
    print(f"DB đã load, có {count} vector.")

# Tạo retriever từ vectorstore
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

llm = Ollama(model="llama3.2:3b", temperature=0.7)
system_prompt = """
Bạn là một trợ lý AI thân thiện, luôn trả lời bằng tiếng Việt, tự nhiên và dễ hiểu. Bạn là người hỗ trợ cung cấp thông tin về
mọi mặt của Trường Đại Học Cần Thơ (CTU) dựa trên tài liệu được cung cấp, sử dụng thông tin trong tài liệu nếu có,
nếu không hãy nói bạn không tìm thấy thông tin phù hợp.
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

    print("=== BẮT ĐẦU TRUY VẤN ===")
    relevant_docs = retriever.invoke(prompt)
    print(f"Số tài liệu tìm thấy: {len(relevant_docs)}")

    if not relevant_docs:
        st.warning("Không tìm thấy tài liệu liên quan, có thể context trống hoặc embedding lỗi.")
    else:
        for i, doc in enumerate(relevant_docs):
            print(f"- Tài liệu {i+1}: {doc.metadata.get('source', 'Không có tên')} | {len(doc.page_content)} ký tự")

    context_documents_str = "\n\n".join(doc.page_content for doc in relevant_docs)

    history_text = ""
    for msg in st.session_state.messages[-6:]:
        role = "Người dùng" if isinstance(msg, HumanMessage) else "Trợ lý"
        history_text += f"{role}: {msg.content}\n"

    qa_prompt_local = qa_prompt.partial(
        history=history_text,
        context=context_documents_str
    )

    llm_chain = {"input": RunnablePassthrough()} | qa_prompt_local | llm
    print("Gọi mô hình LLM...")
    result = llm_chain.invoke(prompt)
    print("Hoàn tất phản hồi từ mô hình.")

    with st.chat_message("assistant"):
        st.markdown(result)
    st.session_state.messages.append(AIMessage(content=result))
