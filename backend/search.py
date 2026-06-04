# search.py

import os

def search_modules(query):
    folder = "backend/data/subjects/cndc/extracted modules"

    results = []

    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        if query.lower() in text.lower():
            results.append(file)

    return results