import os
import sys
import json
from datetime import datetime

from search import search_modules
from llm import ask_gemini

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUBJECTS_DIR = os.path.join(BASE_DIR, "data", "subjects")
LOG_PATH = os.path.join(BASE_DIR, "logs", "questions.log")


def list_subjects():
    if not os.path.isdir(SUBJECTS_DIR):
        return []
    return sorted(
        name for name in os.listdir(SUBJECTS_DIR)
        if os.path.isdir(os.path.join(SUBJECTS_DIR, name))
    )


def load_metadata(subject):
    path = os.path.join(SUBJECTS_DIR, subject, "metadata.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def choose_subject():
    subjects = list_subjects()
    if not subjects:
        print(f"No subjects found under {SUBJECTS_DIR}.")
        sys.exit(1)
    if len(subjects) == 1:
        return subjects[0]

    print("Available subjects:", ", ".join(subjects))
    choice = input(f"Subject [{subjects[0]}]: ").strip().lower()
    return choice if choice in subjects else subjects[0]


def log_question(question, results, subject):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n[{datetime.now()}] subject={subject}\n")
        f.write(f"Question: {question}\n")
        f.write("Sources:\n")
        for item in results:
            chunk = item.get("chunk_index")
            label = f"{item['file']}#{chunk}" if chunk is not None else item["file"]
            f.write(f"- {label} (score: {item['score']:.3f})\n")
        f.write("-" * 50 + "\n")


def build_prompt(question, results, metadata):
    subject_name = metadata.get("subject", "this course")
    teacher = metadata.get("teacher")
    instructor_line = f"This course is taught by {teacher}.\n" if teacher else ""
    context = "\n\n".join(item["text"] for item in results)

    return f"""
You are a teaching assistant for {subject_name} helping engineering students.
{instructor_line}
Use the provided study material as the primary source of truth.

Your job is to teach, not just repeat notes.

Rules:
- Explain concepts in simple, student-friendly language.
- Teach the concept directly as if you are a professor explaining it in class.
- If asked, teach as if you are a peer explaining to another student.
- Use the study material as the primary source of truth.
- You may add simple examples, intuition, applications, and real-world relevance when they help understanding.
- Stay within the scope of {subject_name} and closely related concepts.
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


def main():
    subject = choose_subject()
    metadata = load_metadata(subject)

    question = input("Ask: ").strip()
    if not question:
        print("No question entered.")
        return

    try:
        results = search_modules(question, subject=subject)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    if not results:
        print("No relevant study material found for that question.")
        log_question(question, results, subject)
        return

    log_question(question, results, subject)
    prompt = build_prompt(question, results, metadata)

    try:
        answer = ask_gemini(prompt)
    except RuntimeError as e:
        print(f"Error: {e}")
        return

    print(answer)
    print("\nSources Used:")
    for item in results:
        chunk = item.get("chunk_index")
        label = f"{item['file']}#{chunk}" if chunk is not None else item["file"]
        print(f"- {label} (score: {item['score']:.3f})")


if __name__ == "__main__":
    main()
