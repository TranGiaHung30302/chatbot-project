import streamlit as st

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


st.set_page_config(page_title="Chatbot")
st.title("Chatbot Interface")

#Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append(
        SystemMessage(content='Bạn là một trợ lý AI thân thiện, luôn trả lời hoàn toàn bằng tiếng Việt, diễn đạt tự nhiên và dễ hiểu.')
    )

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

        st.session_state.messages.append({"role": "user", "content": prompt})

    llm = ChatOllama(model="llama3.2:3b", temperature=2)

    result = llm.invoke(st.session_state.messages).content

    with st.chat_message("assistant"):
        st.markdown(result)

        st.session_state.messages.append({"role": "assistant", "content": result})