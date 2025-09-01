import datetime
import os
import sys

import plotly.graph_objects as go
import requests
import streamlit as st

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)

from config_utils import get_api_config, get_quality_config


api_config = get_api_config()
quality_config = get_quality_config()

API_BASE_URL = os.getenv("API_BASE_URL", api_config["base_url"])


st.set_page_config(
    page_title="CodeBot Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


def call_api_chat(message: str):
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json={"message": message},
            headers={"Content-Type": "application/json"},
            timeout=30
        )

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to API: {e}")
        return None


def initialize_session():
    if 'messages' not in st.session_state:
        st.session_state.messages = []


def display_metrics_sidebar():
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Metrics")

    try:
        response = requests.get(f"{API_BASE_URL}/metrics")
        if response.status_code == 200:
            metrics_data = response.json()
            metrics_summary = metrics_data.get("metrics", {})
        else:
            st.sidebar.error(f"Failed to fetch metrics: {response.status_code}")
            metrics_summary = {'total_interactions': 0}
    except Exception as e:
        st.sidebar.error(f"Error getting metrics: {e}")
        metrics_summary = {'total_interactions': 0}

    try:
        response = requests.get(f"{API_BASE_URL}/agents/info")
        if response.status_code == 200:
            agent_info = response.json()
            if agent_info.get("agent_system_active", False):
                st.sidebar.success("🤖 Multi-Agent System: ACTIVE")
                st.sidebar.markdown("**Agent Types:**")
                for agent in agent_info.get("available_agents", []):
                    st.sidebar.markdown(f"- {agent['icon']} {agent['name']}")
            else:
                st.sidebar.warning("⚠️ Agent System: INACTIVE")
        else:
            st.sidebar.warning("🔄 Agent System: UNKNOWN")
    except Exception as e:
        st.sidebar.warning(f"🔄 Agent System: ERROR - {e}")

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
                response = requests.post(f"{API_BASE_URL}/metrics/export")
                if response.status_code == 200:
                    result = response.json()
                    st.sidebar.success(f"Metrics exported: {result.get('message', 'Success')}")
                    st.session_state.exported_metrics = True
                else:
                    st.sidebar.error(f"Export failed: {response.status_code}")
            except Exception as e:
                st.sidebar.error(f"Export failed: {e}")

        if st.session_state.get('exported_metrics', False):
            if st.sidebar.button("🗑️ Clear Exported Metrics"):
                try:
                    response = requests.delete(f"{API_BASE_URL}/metrics")
                    if response.status_code == 200:
                        st.session_state.exported_metrics = False
                        st.sidebar.success("Metrics cleared")
                    else:
                        st.sidebar.error(f"Clear failed: {response.status_code}")
                except Exception as e:
                    st.sidebar.error(f"Clear failed: {e}")
                st.sidebar.success("Exported metrics cleared!")
                st.rerun()

        if st.sidebar.button("Clear metrics"):
            try:
                response = requests.delete(f"{API_BASE_URL}/metrics")
                if response.status_code == 200:
                    st.session_state.exported_metrics = False
                    st.sidebar.success("All metrics cleared!")
                    st.rerun()
                else:
                    st.sidebar.error(f"Clear failed: {response.status_code}")
            except Exception as e:
                st.sidebar.error(f"Clear failed: {e}")
    else:
        st.sidebar.info("No metrics available yet. Start chatting to generate metrics!")


def main():
    st.title("🤖 CodeBot Assistant")

    initialize_session()

    with st.sidebar:
        st.header("Settings")

        try:
            response = requests.get(f"{API_BASE_URL}/")
            if response.status_code == 200:
                data = response.json()
                st.info(data.get("model_info", "CodeBot API connected"))
            else:
                st.warning("API connection issue")
        except Exception as e:
            st.error(f"Failed to connect to API: {e}")

        try:
            response = requests.get(f"{API_BASE_URL}/health")
            if response.status_code == 200:
                health_data = response.json()
                if health_data.get("rag_enabled", False):
                    st.success("RAG knowledge base: Enabled")
                else:
                    st.warning("RAG knowledge base: Disabled")
            else:
                st.warning("RAG knowledge base: Unknown")
        except Exception:
            st.warning("RAG knowledge base: Error")

        if st.button("Clear chat history", type="secondary"):
            try:
                response = requests.delete(f"{API_BASE_URL}/history")
                if response.status_code == 200:
                    st.session_state.messages = []
                    st.rerun()
                else:
                    st.error("Failed to clear conversation")
            except Exception as e:
                st.error(f"Error clearing conversation: {e}")

        display_metrics_sidebar()

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
                    height: 1000px;
                    margin: 0 auto;
                    width: 1px;
                "></div>
                """,
                unsafe_allow_html=True
            )

        with metrics_col:
            st.subheader("Live metrics")
            display_detailed_metrics_compact_api()
    else:
        display_chat_interface()


def display_chat_interface():
    chat_height = 800 if st.session_state.get('show_detailed_metrics', False) else 580
    chat_container = st.container(height=chat_height)

    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "assistant":
                    agents_used = message.get("agents_used", [])
                    intent = message.get("intent", "")
                    quality_score = message.get("quality_score", 0.0)
                    agent_steps = message.get("agent_steps", [])

                    if agents_used:
                        agent_icons = {
                            "router": "🎯", "research": "📚", "code": "💻",
                            "reviewer": "✅", "direct": "🔄"
                        }
                        agent_display = " → ".join([f"{agent_icons.get(agent, '🔧')} {agent.title()}" for agent in agents_used])

                        col1, col2, col3 = st.columns([3, 1, 1])
                        with col1:
                            st.caption(f"🤖 Agent Path: {agent_display}")
                        with col2:
                            if intent:
                                st.caption(f"🎯 Intent: {intent}")
                        with col3:
                            if quality_score > 0:
                                excellent_threshold = quality_config["excellent_threshold"]
                                good_threshold = quality_config["good_threshold"]
                                quality_color = "🟢" if quality_score >= excellent_threshold else "🟡" if quality_score >= good_threshold else "🔴"
                                st.caption(f"{quality_color} Quality: {quality_score:.1f}")

                        if agent_steps:
                            with st.expander("🔍 View Agent Steps", expanded=False):
                                for step in agent_steps:
                                    st.markdown(f"- {step}")
                    elif message.get("rag_used", False):
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
                try:
                    response = requests.get(f"{API_BASE_URL}/agents/info")
                    use_agents = response.status_code == 200 and response.json().get("agent_system_active", False)
                except Exception:
                    use_agents = False

                if use_agents:
                    status_placeholder = st.empty()
                    steps_placeholder = st.empty()
                    response_placeholder = st.empty()

                    _ = []

                    try:
                        status_placeholder.info("🚀 Starting Multi-Agent Processing...")

                        class StreamlitAgentProcessor:
                            def __init__(self):
                                self.steps = []

                            def process_query_with_steps(self, query):
                                self.steps.append("🎯 Router Agent: Analyzing query intent...")
                                steps_placeholder.markdown("**Agent Steps:**\n" + "\n".join([f"- {step}" for step in self.steps]))

                                api_result = call_api_chat(query)

                                if not api_result:
                                    return {
                                        "response": "Failed to process query via API",
                                        "agents_used": ["error"],
                                        "intent": "error",
                                        "quality_score": 0.0
                                    }

                                result = {
                                    "response": api_result.get("response", ""),
                                    "agents_used": api_result.get("agents_used", []),
                                    "intent": api_result.get("intent", ""),
                                    "quality_score": api_result.get("quality_score", 0.0),
                                    "research_results": api_result.get("research_results", []),
                                    "metadata": api_result.get("metadata", {})
                                }

                                agents_used = result.get("agents_used", [])
                                intent = result.get("intent", "")

                                self.steps = []
                                agent_icons = {"router": "🎯", "research": "📚", "code": "💻", "reviewer": "✅", "direct": "🔄"}

                                for i, agent in enumerate(agents_used):
                                    icon = agent_icons.get(agent, "🔧")
                                    if agent == "router":
                                        self.steps.append(f"{icon} Router Agent: Intent classified as {intent}")
                                    elif agent == "research":
                                        self.steps.append(f"{icon} Research Agent: Searching documentation...")
                                        self.steps.append("📖 Research Agent: Found relevant information")
                                    elif agent == "code":
                                        self.steps.append(f"{icon} Code Agent: Generating code solution...")
                                        self.steps.append("✨ Code Agent: Code generated and formatted")
                                    elif agent == "reviewer":
                                        self.steps.append(f"{icon} Reviewer Agent: Evaluating response quality...")
                                        quality = result.get("quality_score", 0)
                                        quality_emoji = "🟢" if quality >= 4 else "🟡" if quality >= 3 else "🔴"
                                        self.steps.append(f"{quality_emoji} Reviewer Agent: Quality score: {quality:.1f}")
                                    elif agent == "direct":
                                        self.steps.append(f"{icon} Direct Response: Generating answer...")

                                    steps_placeholder.markdown("**Agent Steps:**\n" + "\n".join([f"- {step}" for step in self.steps]))

                                self.steps.append("✅ Processing complete!")
                                steps_placeholder.markdown("**Agent Steps:**\n" + "\n".join([f"- {step}" for step in self.steps]))

                                return result

                        wrapper = StreamlitAgentProcessor()
                        result = wrapper.process_query_with_steps(st.session_state.current_prompt)

                        response = result.get("response", "")
                        agents_used = result.get("agents_used", [])
                        intent = result.get("intent", "")
                        quality_score = result.get("quality_score", 0.0)

                        status_placeholder.empty()

                        if response:
                            response_placeholder.markdown(response)

                            message_data = {
                                "role": "assistant",
                                "content": response,
                                "rag_used": "research" in agents_used,
                                "agents_used": agents_used,
                                "intent": intent,
                                "quality_score": quality_score,
                                "agent_steps": wrapper.steps
                            }
                        else:
                            error_msg = "Failed to get response. Please try again!"
                            response_placeholder.error(error_msg)
                            message_data = {
                                "role": "assistant",
                                "content": error_msg,
                                "rag_used": False
                            }

                    except Exception as e:
                        status_placeholder.empty()
                        steps_placeholder.empty()
                        error_msg = f"Error occurred: {e}"
                        response_placeholder.error(error_msg)
                        message_data = {
                            "role": "assistant",
                            "content": error_msg,
                            "rag_used": False
                        }
                else:
                    with st.spinner("Thinking..."):
                        try:
                            api_response = call_api_chat(st.session_state.current_prompt)

                            if api_response and api_response.get("success"):
                                response = api_response.get("response", "")
                                rag_used = api_response.get("rag_used", False)

                                message_data = {
                                    "role": "assistant",
                                    "content": response,
                                    "rag_used": rag_used,
                                    "agents_used": api_response.get("agents_used", []),
                                    "quality_score": api_response.get("quality_score", 0),
                                    "intent": api_response.get("intent", "")
                                }
                            else:
                                error_msg = "Failed to get response. Please try again!"
                                message_data = {
                                    "role": "assistant",
                                    "content": error_msg,
                                    "rag_used": False
                                }

                        except Exception as e:
                            error_msg = f"Error occurred: {e}"
                            message_data = {
                                "role": "assistant",
                                "content": error_msg,
                                "rag_used": False
                            }

                st.session_state.messages.append(message_data)
                st.session_state.processing_response = False
                if 'current_prompt' in st.session_state:
                    del st.session_state.current_prompt

                st.rerun()

    if prompt := st.chat_input("Ask me anything about programming in python..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        st.session_state.processing_response = True
        st.session_state.current_prompt = prompt
        st.rerun()


def display_detailed_metrics_compact_api():
    try:
        response = requests.get(f"{API_BASE_URL}/metrics/detailed")
        if response.status_code == 200:
            data = response.json()
            metrics = data.get("detailed_metrics", {})

            total_interactions = metrics.get("total_interactions", 0)

            if total_interactions == 0:
                st.info("No interactions recorded yet.")
                return

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Interactions", total_interactions)
                st.metric("RAG usage", f"{metrics.get('rag_usage_rate', 0):.1%}")
            with col2:
                st.metric("Avg time", f"{metrics.get('avg_response_time', 0):.1f}s")
                st.metric("Quality", f"{metrics.get('avg_quality_score', 0):.1f}/5")

            session_start = metrics.get("session_start_time")
            if session_start:
                start_time = datetime.datetime.fromtimestamp(session_start)
                current_time = datetime.datetime.now()
                session_duration = (current_time - start_time).total_seconds()
                st.metric("Session duration", f"{session_duration:.0f}s")
            else:
                st.metric("Session duration", f"{metrics.get('total_session_time', 0):.1f}s")

            latest_interaction = metrics.get("latest_interaction", {})
            if latest_interaction:
                st.markdown("**Latest response:**")
                col3, col4 = st.columns(2)
                with col3:
                    if 'response_word_count' in latest_interaction:
                        st.metric("Words", latest_interaction['response_word_count'])
                    if 'agents_used' in latest_interaction and latest_interaction['agents_used']:
                        agent_count = len([a for a in latest_interaction['agents_used'] if a.strip()])
                        st.metric("Agents used", agent_count)
                with col4:
                    if 'perplexity_approx' in latest_interaction:
                        st.metric("Perplexity", f"{latest_interaction['perplexity_approx']:.1f}")
                    if 'quality_score' in latest_interaction:
                        st.metric("Quality", f"{latest_interaction['quality_score']:.1f}/5")

            response_times = metrics.get("response_time_history", [])
            if len(response_times) > 1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=response_times[-10:],
                    mode='lines+markers',
                    name='Response Time',
                    line=dict(color='#1f77b4', width=2),
                    marker=dict(size=6)
                ))
                fig.update_layout(
                    title="Response time trend (last 10)",
                    height=200,
                    margin=dict(l=0, r=0, t=30, b=20),
                    showlegend=False,
                    xaxis_title="Interaction",
                    yaxis_title="Time (s)"
                )
                st.plotly_chart(fig, use_container_width=True)

            quality_scores = metrics.get("quality_score_history", [])
            if len(quality_scores) > 1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=quality_scores[-10:],
                    mode='lines+markers',
                    name='Quality Score',
                    line=dict(color='#2ca02c', width=2),
                    marker=dict(size=6)
                ))
                fig.update_layout(
                    title="Quality trend (last 10)",
                    height=200,
                    margin=dict(l=0, r=0, t=30, b=20),
                    showlegend=False,
                    xaxis_title="Interaction",
                    yaxis_title="Quality (1-5)"
                )
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.error(f"Failed to fetch detailed metrics: {response.status_code}")
    except Exception as e:
        st.error(f"Error loading detailed metrics: {e}")


if __name__ == "__main__":
    main()
