# 🎓 Agentic AI Study Coach

An Agentic AI assistant that builds personalized study plans using:

- **LangChain Tools**
- **Groq LLMs**
- **A multi-tool agent with reasoning**
- **A Gradio UI**

Built as an Agentic AI Portfolio Project demonstrating:

✔ Tool calling  
✔ Multi-step reasoning  
✔ Stateful agent logic  
✔ UI integration  
✔ Clean project structure  

---

# 🚀 Features

### 🔧 Agent Tools
The agent uses 3 custom tools:
1. **Module Outline Tool**  
2. **Study Plan Generator Tool**  
3. **Practice Task Generator Tool**

### 🧠 LLM Reasoning  
Uses Groq (Llama 3.1 8B) via LangChain.

### 🖥 UI  
Interactive **Gradio** web interface showing:
- Final answer  
- Sequence of tool calls  
- High-level reasoning
  
# 📂 Project Structure
agentic-ai-study-coach/
│
├── app/
│ ├── agent.py # LangChain agent + tools
│ └── appui.py # Gradio web interface
│
├── requirements.txt # Python dependencies
└── README.md # Project documentation


---

## ▶️ Running the Project

# 1. Clone the repository
git clone https://github.com/abdullahsherazi-sketch/agentic-ai-study-coach
cd agentic-ai-study-coach

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your Groq API key

# Mac / Linux
export GROQ_API_KEY="your_key_here"

# Windows (PowerShell)
setx GROQ_API_KEY "your_key_here"

# 4. Run the Gradio UI
python app/appui.py
