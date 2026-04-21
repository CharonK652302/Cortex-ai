"""
Cortex AI — Crew.ai Multi-Agent Research Pipeline
===================================================
Place this in the ROOT of your Cortex AI repo.
Two agents: Researcher + Writer working sequentially.

Resume bullet (new project line):
  "Built Crew.ai multi-agent pipeline — Researcher +
   Writer agents with sequential task delegation"

Run:
  python crewai_pipeline.py "RAG in LLMs"
  python crewai_pipeline.py "LangChain vs LlamaIndex"
"""

import sys
from crewai import Agent, Task, Crew, Process


# ── Agents ────────────────────────────────────────────────────────────────────

researcher = Agent(
    role="Senior AI Research Analyst",
    goal=(
        "Research the given topic and extract key facts, "
        "recent developments, and actionable insights. "
        "Be concise and structured."
    ),
    backstory=(
        "You are an expert AI researcher with deep knowledge of "
        "machine learning, LLMs, and GenAI systems. You distill "
        "complex topics into clear, structured findings."
    ),
    verbose=True,
    allow_delegation=False,
    max_iter=2,
)

writer = Agent(
    role="Technical Report Writer",
    goal=(
        "Transform research findings into a professional, "
        "well-structured report with clear sections."
    ),
    backstory=(
        "You are a technical writer specializing in AI. "
        "You turn raw research into polished reports that "
        "developers and hiring managers love to read."
    ),
    verbose=True,
    allow_delegation=False,
    max_iter=2,
)


# ── Tasks ─────────────────────────────────────────────────────────────────────

def build_tasks(topic: str):
    research_task = Task(
        description=(
            f"Research this topic thoroughly: '{topic}'\n\n"
            "Cover:\n"
            "1. What it is and why it matters\n"
            "2. Key technical components\n"
            "3. Real-world use cases\n"
            "4. Current limitations\n"
            "5. Future direction\n\n"
            "Output: 10-15 structured bullet points."
        ),
        expected_output=(
            "A structured list of 10-15 bullet points covering "
            "all 5 sections about the topic."
        ),
        agent=researcher,
    )

    write_task = Task(
        description=(
            f"Write a professional technical report on: '{topic}'\n\n"
            "Use the research findings. Structure:\n"
            "- Executive Summary (2-3 sentences)\n"
            "- Overview\n"
            "- Technical Details\n"
            "- Use Cases\n"
            "- Limitations\n"
            "- Conclusion\n\n"
            "Keep it under 500 words. Professional tone."
        ),
        expected_output=(
            "A complete technical report under 500 words "
            "with all 6 sections."
        ),
        agent=writer,
        context=[research_task],
    )

    return research_task, write_task


# ── Crew ──────────────────────────────────────────────────────────────────────

def run(topic: str) -> str:
    print(f"\n{'='*55}")
    print(f"CREW.AI PIPELINE — Topic: {topic}")
    print(f"Agents: Researcher → Writer (Sequential)")
    print(f"{'='*55}\n")

    research_task, write_task = build_tasks(topic)

    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, write_task],
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff()
    return str(result)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 \
        else "Retrieval-Augmented Generation (RAG)"

    report = run(topic)

    print(f"\n{'='*55}")
    print("FINAL REPORT")
    print(f"{'='*55}")
    print(report)

    # Save report to file
    fname = f"report_{topic[:30].replace(' ','_')}.md"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(f"# {topic}\n\n{report}")
    print(f"\n[+] Saved to {fname}")
