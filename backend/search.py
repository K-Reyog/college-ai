import os


def search_modules(query):
    folder = "backend/data/subjects/cndc/extracted_modules"

    results = [] #array for the results

    # Break question into words to find matches in the text files
    words = query.lower().split()

    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        text_lower = text.lower()

        # Calculate relevance score
        score = 0
        #if two or more words match, it's more likely to be relevant, so we count the occurrences of each word in the text and add them to the score
        for word in words:
            if len(word) > 2 and word in text_lower:
                score += text_lower.count(word)

        # Ignore completely irrelevant files
        if score > 0:

            # Find the first matching word to extract nearby text for context. If no match is found, we just take the module which has the first match.
            first_match = -1

            for word in words:
                position = text_lower.find(word)

                if position != -1:
                    first_match = position
                    break

            if first_match == -1:
                first_match = 0

            # Extract nearby text
            start = max(0, first_match - 500)
            end = min(len(text), first_match + 2000)

            snippet = text[start:end]

            results.append({
                "file": file,
                "score": score,
                "text": snippet
            })

    # Highest score first
    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:3]