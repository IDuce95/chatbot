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
            st.session_state.chatbot = ChatBot()
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

        if st.button("Clear chat history", type="secondary"):
            st.session_state.chatbot.clear_history()
            st.session_state.messages = []
            st.rerun()

    chat_container = st.container()

    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if prompt := st.chat_input("Ask me anything about programming in python..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            thinking_placeholder = st.empty()
            thinking_placeholder.markdown("**Thinking...**")

            try:
                response = st.session_state.chatbot.get_response(prompt)

                thinking_placeholder.empty()

                if response:
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    error_msg = "Failed to get response. Please try again!"
                    st.markdown(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

            except Exception as e:
                thinking_placeholder.empty()
                error_msg = f"Error occurred: {e}"
                st.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})


if __name__ == "__main__":
    main()
