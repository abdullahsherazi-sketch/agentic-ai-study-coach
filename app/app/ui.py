"""
Gradio UI for the Agentic AI Study Coach Project.

This UI loads:
- The Study Coach agent (Groq LLM + LangChain Tools)
- A textbox UI for student queries
- A clean display of the final answer + tool call log
"""

import os
import gradio as gr

from agent import build_agent, run_study_coach

# -------------------------------------------------
# Load agent once at startup
# -------------------------------------------------
agent = None

def load_agent():
    global agent

    if agent is None:
        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError(
                "GROQ_API_KEY environment variable is not set.\n"
                "Please set it before running the UI:\n"
                "   export GROQ_API_KEY='your-key-here'"
            )

        agent = build_agent()
    return agent


# -------------------------------------------------
# Main function called by Gradio
# -------------------------------------------------
def study_coach_interface(query: str):
    try:
        load_agent()
        final_answer, tool_log = run_study_coach(agent, query)
        return final_answer, tool_log
    except Exception as e:
        return f"❌ Error: {str(e)}", ""


# -------------------------------------------------
# Build Gradio UI
# -------------------------------------------------
with gr.Blocks(title="Study Coach Agent") as demo:
    gr.Markdown(
        """
        # 🎓 Study Coach Agent  
        An Agentic AI assistant that builds personalized study plans  
        using **LangChain Tools**
