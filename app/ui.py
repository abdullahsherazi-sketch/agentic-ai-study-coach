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
        using **LangChain Tools**, **Groq LLMs**, and a **Gradio UI**.
        """
    )

    with gr.Row():
        query_box = gr.Textbox(
            label="Ask the Study Coach",
            placeholder=(
                "Example: I have a Generative AI exam in 5 days, 3 hours/day. "
                "Weak in LangChain basics and RAG."
            ),
            lines=3,
        )

    submit_button = gr.Button("Submit", variant="primary")
    clear_button = gr.Button("Clear")

    with gr.Row():
        response_box = gr.Markdown(label="Study Coach Response")
        tool_log_box = gr.Textbox(label="Tool Call Log", lines=8)

    submit_button.click(
        study_coach_interface,
        inputs=query_box,
        outputs=[response_box, tool_log_box],
    )

    clear_button.click(
        lambda: ("", ""),
        None,
        [response_box, tool_log_box],
    )


# -------------------------------------------------
# Launch the UI
# -------------------------------------------------
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
