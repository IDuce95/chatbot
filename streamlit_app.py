import os
import sys

import plotly.graph_objects as go
import streamlit as st

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


def display_metrics_sidebar(chatbot):
    st.sidebar.markdown("---")
    st.sidebar.header("Metrics")

    metrics_summary = chatbot.get_metrics_summary()

    if metrics_summary.get('total_interactions', 0) > 0:
        col1, col2 = st.sidebar.columns(2)

        with col1:
            st.metric("Total interactions", metrics_summary.get('total_interactions', 0))
            st.metric("RAG usage", f"{metrics_summary.get('rag_usage_rate', 0):.1%}")

        with col2:
            st.metric("Avg response time", f"{metrics_summary.get('avg_response_time', 0):.2f}s")
            st.metric("Total session time", f"{metrics_summary.get('total_session_time', 0):.1f}s")

        if st.sidebar.button("View detailed metrics"):
            st.session_state.show_detailed_metrics = not st.session_state.get('show_detailed_metrics', False)
            st.rerun()

        if st.sidebar.button("Export metrics"):
            try:
                filename = "streamlit_metrics_export.json"
                chatbot.export_metrics(filename)
                st.sidebar.success(f"Metrics exported to {filename}")
                st.session_state.exported_metrics = True
            except Exception as e:
                st.sidebar.error(f"Export failed: {e}")

        if st.session_state.get('exported_metrics', False):
            if st.sidebar.button("🗑️ Clear Exported Metrics"):
                chatbot.clear_metrics()
                st.session_state.exported_metrics = False
                st.sidebar.success("Exported metrics cleared!")
                st.rerun()

        if st.sidebar.button("Clear metrics"):
            chatbot.clear_metrics()
            st.session_state.exported_metrics = False
            st.sidebar.success("All metrics cleared!")
            st.rerun()
    else:
        st.sidebar.info("No metrics available yet. Start chatting to generate metrics!")


def main():
    st.title("🤖 CodeBot Assistant")

    initialize_chatbot()

    with st.sidebar:
        st.header("Settings")

        st.info(st.session_state.chatbot.get_model_info())

        if st.session_state.chatbot.use_rag:
            st.success("RAG knowledge base: Enabled")
        else:
            st.warning("RAG knowledge base: Disabled")

        if st.button("Clear chat history", type="secondary"):
            st.session_state.chatbot.clear_history()
            st.session_state.messages = []
            st.rerun()

        display_metrics_sidebar(st.session_state.chatbot)

    if st.session_state.get('show_detailed_metrics', False):
        chat_col, separator_col, metrics_col = st.columns([2, 0.1, 1])

        with chat_col:
            st.subheader("Chat")
            display_chat_interface()

        with separator_col:
            st.markdown(
                """
                <div style="
                    border-left: 2px solid #e0e0e0;
                    height: 900px;
                    margin: 0 auto;
                    width: 1px;
                "></div>
                """,
                unsafe_allow_html=True
            )

        with metrics_col:
            st.subheader("Live metrics")
            display_detailed_metrics_compact(st.session_state.chatbot)
    else:
        display_chat_interface()


def display_chat_interface():
    chat_height = 800 if st.session_state.get('show_detailed_metrics', False) else 580
    chat_container = st.container(height=chat_height)

    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "assistant":
                    if message.get("rag_used", False):
                        st.caption("🧠 Enhanced with knowledge base")

                    gen_metrics = message.get("generation_metrics", {})
                    if gen_metrics and not st.session_state.get('show_detailed_metrics', False):
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.markdown(message["content"])
                        with col2:
                            with st.expander("📊 Metrics"):
                                if 'response_word_count' in gen_metrics:
                                    st.metric("Words", gen_metrics['response_word_count'])
                                if 'perplexity_approx' in gen_metrics:
                                    st.metric("Perplexity", f"{gen_metrics['perplexity_approx']:.1f}")
                                if 'unique_word_ratio' in gen_metrics:
                                    st.metric("Uniqueness", f"{gen_metrics['unique_word_ratio']:.2f}")
                    else:
                        st.markdown(message["content"])
                else:
                    st.markdown(message["content"])

        if st.session_state.get('processing_response', False):
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.chatbot.get_response(st.session_state.current_prompt)

                        if response:
                            rag_used = hasattr(st.session_state.chatbot, 'last_rag_used') and st.session_state.chatbot.last_rag_used

                            gen_metrics = {}
                            if hasattr(st.session_state.chatbot, 'rag_manager') and st.session_state.chatbot.rag_manager:
                                interactions = st.session_state.chatbot.rag_manager.metrics.session_metrics
                                if interactions:
                                    gen_metrics = interactions[-1].get('generation_metrics', {})

                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response,
                                "rag_used": rag_used,
                                "generation_metrics": gen_metrics
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

    if prompt := st.chat_input("Ask me anything about programming in python..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        st.session_state.processing_response = True
        st.session_state.current_prompt = prompt
        st.rerun()


def display_detailed_metrics_compact(chatbot):
    if hasattr(chatbot, 'rag_manager') and chatbot.rag_manager:
        interactions = chatbot.rag_manager.metrics.session_metrics

        if not interactions:
            st.info("No interactions recorded yet.")
            return

        total_interactions = len(interactions)
        rag_interactions = sum(1 for i in interactions if i.get('rag_used', False))
        avg_response_time = sum(i.get('response_time', 0) for i in interactions) / len(interactions)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Interactions", total_interactions)
            st.metric("RAG usage", f"{rag_interactions}/{total_interactions}")
        with col2:
            st.metric("Avg time", f"{avg_response_time:.1f}s")
            if interactions:
                total_time = interactions[-1]['timestamp'] - interactions[0]['timestamp']
                st.metric("Session", f"{total_time:.0f}s")

        if interactions:
            latest = interactions[-1]
            gen_metrics = latest.get('generation_metrics', {})

            st.markdown("**Latest response:**")
            if 'response_word_count' in gen_metrics:
                st.metric("Words", gen_metrics['response_word_count'])
            if 'perplexity_approx' in gen_metrics:
                st.metric("Perplexity", f"{gen_metrics['perplexity_approx']:.1f}")
            if 'unique_word_ratio' in gen_metrics:
                st.metric("Uniqueness", f"{gen_metrics['unique_word_ratio']:.2f}")

            retrieval_metrics = latest.get('retrieval_metrics', {})
            auto_metrics = {k: v for k, v in retrieval_metrics.items() if 'auto_' in k}
            if auto_metrics:
                st.markdown("**RAG quality:**")
                if 'auto_precision@1' in auto_metrics:
                    st.metric("Precision@1", f"{auto_metrics['auto_precision@1']:.2f}")
                if 'auto_relevance_rate' in auto_metrics:
                    st.metric("Relevance rate", f"{auto_metrics['auto_relevance_rate']:.2f}")

        if len(interactions) > 1:
            response_times = [i.get('response_time', 0) for i in interactions[-10:]]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=response_times,
                mode='lines+markers',
                name='Response Time',
                line=dict(color='#1f77b4', width=2)
            ))
            fig.update_layout(
                title="Response time trend",
                height=200,
                margin=dict(l=0, r=0, t=30, b=0),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
