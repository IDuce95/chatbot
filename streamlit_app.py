import streamlit as st
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from chatbot import ChatBot


st.set_page_config(
    page_title="CodeBot Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


def initialize_chatbot():
    if 'chatbot' not in st.session_state:
        try:
            st.session_state.chatbot = ChatBot(use_rag=True)
            st.session_state.messages = []
        except Exception as e:
            st.error(f"Failed to initialize ChatBot: {e}")
            st.stop()


def main():
    st.title("🤖 CodeBot Assistant")

    initialize_chatbot()

    with st.sidebar:
        st.header("Settings")

        st.info(st.session_state.chatbot.get_model_info())

        if st.session_state.chatbot.use_rag:
            st.success("RAG Knowledge Base: Enabled")
        else:
            st.warning("RAG Knowledge Base: Disabled")

        if st.button("Clear chat history", type="secondary"):
            st.session_state.chatbot.clear_history()
            st.session_state.messages = []
            st.rerun()

    chat_container = st.container()

    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "assistant" and message.get("rag_used", False):
                    st.caption("🧠 Enhanced with knowledge base")
                st.markdown(message["content"])

    if prompt := st.chat_input("Ask me anything about programming in python..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        st.session_state.processing_response = True
        st.session_state.current_prompt = prompt
        st.rerun()

    if st.session_state.get('processing_response', False):
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.chatbot.get_response(st.session_state.current_prompt)

                    if response:
                        rag_used = hasattr(st.session_state.chatbot, 'last_rag_used') and st.session_state.chatbot.last_rag_used

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response,
                            "rag_used": rag_used
                        })
                    else:
                        error_msg = "Failed to get response. Please try again!"
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg,
                            "rag_used": False
                        })

                except Exception as e:
                    error_msg = f"Error occurred: {e}"
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "rag_used": False
                    })

                st.session_state.processing_response = False
                if 'current_prompt' in st.session_state:
                    del st.session_state.current_prompt

                st.rerun()


if __name__ == "__main__":
    main()
