"""
Agent and tools for the Study Coach project.

This module defines:
- SYLLABUS data
- Three LangChain tools
- A Groq-powered Study Coach agent
- A helper function to run the agent and collect a tool-call log
"""

import os
from typing import List, Tuple

from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.tools import tool

# --------------------------------------------------------------------
# Global tool log
# --------------------------------------------------------------------
TOOL_LOG: List[dict] = []


def log_tool(name: str, **kwargs) -> None:
    """Append a tool call entry to the global TOOL_LOG."""
    TOOL_LOG.append({"tool": name, "input": kwargs})


# --------------------------------------------------------------------
# Syllabus & practice-task data
# --------------------------------------------------------------------
SYLLABUS = {
    "generative ai": [
        "LLM fundamentals",
        "Prompt engineering",
        "OpenAI & Groq APIs",
        "LangChain basics",
        "Agents and Tools",
        "RAG",
        "Agent security & guardrails",
    ]
}

PRACTICE_TASKS = {
    "LLM fundamentals": [
        "Explain in your own words what an LLM is and how it is trained.",
        "Compare two LLM architectures and list their pros/cons.",
    ],
    "Prompt engineering": [
        "Write prompts in zero-shot, few-shot and chain-of-thought styles.",
        "Rewrite a vague prompt into a precise, constrained one.",
    ],
    "LangChain basics": [
        "Build a simple LangChain LLMChain using a PromptTemplate.",
        "Create a LangChain chain that calls two steps in sequence.",
    ],
    "Agents and Tools": [
        "Create a ReAct agent that uses at least two tools.",
        "Implement a custom LangChain tool and expose it to an agent.",
    ],
    "RAG": [
        "Implement a basic RAG pipeline using a single PDF.",
        "Experiment with different chunk sizes and compare answer quality.",
    ],
    "Agent security & guardrails": [
        "Design a simple prompt-based safety policy for your agent.",
        "Add checks to block obviously unsafe or irrelevant queries.",
    ],
}


# --------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------
@tool
def get_module_outline(module_name: str) -> str:
    """
    Return the list of topics covered in the specified module.

    This helps the agent understand what the student needs to study.
    """
    log_tool("get_module_outline", module_name=module_name)

    topics = SYLLABUS.get(module_name.lower().strip())
    if not topics:
        return f"No module named '{module_name}' found."

    lines = [f"Topics for {module_name}:"]
    lines += [f"- {t}" for t in topics]
    return "\n".join(lines)


@tool
def build_study_schedule(
    module_name: str,
    days_until_exam: int,
    hours_per_day: float,
    weak_topics: str = "",
) -> str:
    """
    Build a day-by-day study schedule for the student.

    It prioritizes weak topics while still covering the full syllabus.
    """
    log_tool(
        "build_study_schedule",
        module_name=module_name,
        days_until_exam=days_until_exam,
        hours_per_day=hours_per_day,
        weak_topics=weak_topics,
    )

    topics = SYLLABUS.get(module_name.lower().strip())
    if not topics:
        return f"No module named '{module_name}' found."

    weak_list = [w.strip().lower() for w in weak_topics.split(",") if w.strip()]

    # Give weak topics double weight
    weights = [
        2 if any(w in t.lower() for w in weak_list) else 1 for t in topics
    ]
    total_weight = sum(weights)
    total_hours = days_until_exam * hours_per_day
    hours_per_topic = [(w / total_weight) * total_hours for w in weights]

    # Greedy distribution of topic hours across days
    schedule = {day: [] for day in range(1, days_until_exam + 1)}
    day = 1
    for topic, hours in zip(topics, hours_per_topic):
        remaining = hours
        while remaining > 0 and day <= days_until_exam:
            # Allocate up to half a day's hours per "chunk"
            chunk = min(remaining, hours_per_day / 2)
            schedule[day].append((topic, round(chunk, 1)))
            remaining -= chunk

            day_load = sum(h for _, h in schedule[day])
            if day_load >= hours_per_day * 0.9:
                day += 1

    lines = [
        f"Study schedule for '{module_name}'",
        f"Days until exam: {days_until_exam}, Hours per day: {hours_per_day}",
        f"Weak topics prioritized: {weak_topics or 'None specified'}",
        "",
    ]
    for d, items in schedule.items():
        if not items:
            continue
        lines.append(f"Day {d}:")
        for topic, hrs in items:
            lines.append(f"  - {topic}: {hrs} hour(s)")
        lines.append("")

    return "\n".join(lines) if len(lines) > 4 else "Could not build a useful schedule."


@tool
def suggest_practice_tasks(module_name: str, focus_topics: str = "") -> str:
    """
    Suggest hands-on practice tasks for the student's weak or focus topics.
    """
    log_tool(
        "suggest_practice_tasks",
        module_name=module_name,
        focus_topics=focus_topics,
    )

    topics = SYLLABUS.get(module_name.lower().strip())
    if not topics:
        return f"No module named '{module_name}' found."

    focus_list = [f.strip().lower() for f in focus_topics.split(",") if f.strip()]

    lines: List[str] = ["Practice tasks:"]
    for topic in topics:
        if focus_list and not any(f in topic.lower() for f in focus_list):
            continue

        tasks = PRACTICE_TASKS.get(topic, [])
        if not tasks:
            continue

        lines.append(f"* {topic}")
        for t in tasks:
            lines.append(f"  - {t}")
        lines.append("")

    if len(lines) == 1:
        return "No specific practice tasks found for the given focus topics."

    return "\n".join(lines)


# --------------------------------------------------------------------
# Agent creation & runner
# --------------------------------------------------------------------
def build_agent() -> object:
    """
    Build and return the Study Coach Agent.

    Requires the GROQ_API_KEY environment variable to be set.
    """
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY environment variable is not set. "
            "Set it before using the agent."
        )

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3,
    )

    tools = [get_module_outline, build_study_schedule, suggest_practice_tasks]

    system_prompt = """
You are a helpful Study Coach Agent for the 'Generative AI' module.

Your goals:
- Understand the student's exam timeline and weak areas.
- Use tools when helpful:
  - get_module_outline
  - build_study_schedule
  - suggest_practice_tasks
- Combine tool outputs into a clear, actionable study plan.
- Do NOT talk about internal tool calls; focus on advice, schedules and tasks.
"""

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
    )
    return agent


def run_study_coach(agent: object, query: str) -> Tuple[str, str]:
    """
    Run the Study Coach agent on a user query.

    Returns:
        (final_answer, tool_log_text)
    """
    TOOL_LOG.clear()

    result = agent.invoke(
        {"messages": [{"role": "user", "content": query}]}
    )
    final_answer = result["messages"][-1].content

    if TOOL_LOG:
        log_lines = ["Tool call sequence:"]
        for i, entry in enumerate(TOOL_LOG, start=1):
            log_lines.append(f"{i}. {entry['tool']} → {entry['input']}")
        tool_log_text = "\n".join(log_lines)
    else:
        tool_log_text = "No tools were called."

    return final_answer, tool_log_text
