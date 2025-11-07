import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

st.set_page_config(page_title="Chatbot", layout="wide")
st.title("Chatbot")

if "llm" not in st.session_state:
    st.session_state.llm = ChatOllama(model="llama3.2:3b", temperature=0.7)

# --- Khởi tạo ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="Bạn là một trợ lý AI thân thiện, chỉ nói tiếng Việt, trả lời ngắn gọn và rõ ràng.")
    ]

if "waiting_for_reply" not in st.session_state:
    st.session_state.waiting_for_reply = False


chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)


prompt = st.chat_input("Nhập tin nhắn của bạn...")

if prompt:
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.session_state.waiting_for_reply = True
    st.rerun()

# --- Xử lý phản hồi  ---
if st.session_state.waiting_for_reply:
    with st.chat_message("assistant"):
        st.markdown("_Đang trả lời..._")

    llm = ChatOllama(model="llama3.2:3b", temperature=0.7)
    response = llm.invoke(st.session_state.messages).content

    st.session_state.messages.append(AIMessage(content=response))
    st.session_state.waiting_for_reply = False
    st.rerun()