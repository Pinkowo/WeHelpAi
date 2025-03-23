import csv
import re

from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger

ws_driver = CkipWordSegmenter(model="bert-base")
pos_driver = CkipPosTagger(model="bert-base")


def non_punctuation(text):
    return re.sub(r"[，,。：:；;！!？?～~]", "", text)


def tokenize_title(text):
    tokens = re.split(r"\s+", text.strip())
    kept_words = []

    for token in tokens:
        token = token.strip()
        if not token:
            continue
        match = re.match(r"^(.*)\((.*?)\)$", token)
        if match:
            word = match.group(1).strip()
            tag = match.group(2).strip()
            if tag.lower().startswith("p") or tag.lower().startswith("d"):
                continue
            cleaned_word = non_punctuation(word)
            if cleaned_word:
                kept_words.append(cleaned_word)
        else:
            cleaned_token = non_punctuation(token)
            if cleaned_token:
                kept_words.append(cleaned_token)

    return " ".join(kept_words)


def pack_ws_pos_sentence(sentence_ws, sentence_pos):
    assert len(sentence_ws) == len(sentence_pos)
    res = []
    for word_ws, word_pos in zip(sentence_ws, sentence_pos):
        res.append(f"{word_ws}({word_pos})")
    return " ".join(res)


def process_csv_file(filepath):
    cleaned_rows = []
    boards = []
    categories = []
    titles = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            board = row.get("board", "")
            category = row.get("category", "")
            title = row.get("title", "")
            boards.append(board)
            categories.append(category)
            titles.append(title)

    ws_results = ws_driver(titles, batch_size=256)
    pos_results = pos_driver(ws_results)

    for board, sentence_ws, sentence_pos in zip(boards, ws_results, pos_results):
        packed_sentence = pack_ws_pos_sentence(sentence_ws, sentence_pos)
        tokenized_title = tokenize_title(packed_sentence)
        cleaned_rows.append(
            {"board": board, "category": category, "title": tokenized_title}
        )

    return cleaned_rows


def main():
    input_csv = "data-clean.csv"
    output_csv = "data-clean-token.csv"
    cleaned_rows = process_csv_file(input_csv)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["board", "category", "title"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(cleaned_rows)
    print(f"Saved {len(cleaned_rows)} rows to {output_csv}")


if __name__ == "__main__":
    main()
