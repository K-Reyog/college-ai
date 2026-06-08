from search import search_modules
from llm import ask_gemini
from datetime import datetime
# This function logs the question asked by the user along with a timestamp. It appends the log to a file named "questions.log" in the "backend/logs" directory. Each log entry includes the timestamp, the question, and a separator line for clarity.
def log_question(question, results):
    
    timestamp = datetime.now()

    with open(
        "backend/logs/questions.log",
        "a",
        encoding="utf-8"
    ) as f:

        f.write(
            f"\n[{timestamp}]\n"
        )

        f.write(
            f"Question: {question}\n"
        )
        f.write("Sources:\n")

        for item in results:
            f.write(
                f"- {item['file']} "
                f"(score: {item['score']})\n"
            )
        f.write(
            "-" * 50 + "\n"
        )

question = input("Ask: ")

results = search_modules(question)

log_question(question, results)
context = ""

for item in results:
    context += item["text"] + "\n\n"
#here we build a RAG prompt for Gemini. We tell it to only answer using the study material we found, and then we give it the question. This way, we can ensure that Gemini's answer is based on the relevant content we found in the modules.
prompt = f"""
You are a CNDC teaching assistant helping engineering students.

Use the provided study material as the primary source of truth.

Your job is to teach, not just repeat notes.

Rules:
- Explain concepts in simple, student-friendly language.
- Teach the concept directly as if you are a professor explaining it in class.
- If asked, teach as if you are a peer explaining to another student.
- Use the study material as the primary source of truth.
- You may add simple examples, intuition, applications, and real-world relevance when they help understanding.
- Stay within the scope of CNDC and closely related networking concepts.
- Do not introduce advanced topics unless they directly help answer the question.
- Structure answers clearly using headings and bullet points where appropriate.
- For theoretical questions, try to include:
  1. Definition
  2. Working/Explanation
  3. Advantages or Key Features
  4. Applications (if relevant)
  5. Exam-Relevant Points

- Do not refer to "the study material", "the notes", or "the provided material" unless information is missing.

- If the study material contains enough information, answer confidently.

- If the study material is incomplete but the answer can be reasonably inferred from the course context, provide a brief explanation and clearly indicate that additional details are not covered in the modules.

- If the answer cannot be determined from the study material and course context, say so instead of inventing information.

- If the question is ambiguous, ask for clarification.
Study Material:
{context}

Question:
{question}
"""

answer = ask_gemini(prompt)

print(answer)
print("\nSources Used:")

for item in results:
    print(f"- {item['file']} (score: {item['score']})")