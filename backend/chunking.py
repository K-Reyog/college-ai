import re

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def normalize_text(raw_text):
    """Collapse PDF-extraction line-wrap/blank-line noise into clean continuous text."""
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text, chunk_size=1200, overlap=200, lookahead=300):
    """Split normalized text into overlapping chunks snapped to sentence boundaries."""
    chunks = []
    n = len(text)
    if n == 0:
        return chunks

    start = 0
    index = 0

    while start < n:
        end = min(start + chunk_size, n)

        if end < n:
            window = text[end:end + lookahead]
            match = SENTENCE_BOUNDARY.search(window)
            if match:
                end += match.end()
            else:
                space = text.find(" ", end)
                if space != -1 and space - end < lookahead:
                    end = space

        piece = text[start:end].strip()
        if piece:
            chunks.append({"index": index, "text": piece})
            index += 1

        if end >= n:
            break

        start = max(end - overlap, start + 1)

    return chunks
