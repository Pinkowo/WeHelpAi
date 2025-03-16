import csv
import os
import re


def extract_category(title):
    title = title.strip()
    match = re.match(r"^\[(.*?)\]\s*(.*)", title)
    if match:
        category = match.group(1).strip()
        new_title = match.group(2).strip()
        return category, new_title
    else:
        return "", title


def clean_title(title):
    cleaned = title.strip().lower()
    if cleaned.startswith("re:"):
        cleaned = cleaned[len("re:") :].strip()
    if cleaned.startswith("fw:"):
        cleaned = cleaned[len("fw:") :].strip()
    return cleaned


def process_csv_file(filepath):
    cleaned_rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            original_title = row.get("title", "")
            cleaned = clean_title(original_title)
            category, extracted_title = extract_category(cleaned)
            final_title = extracted_title.strip().lower()
            row["category"] = category
            row["title"] = final_title
            cleaned_rows.append(row)
    return cleaned_rows


def combine_csv_files(input_dir, output_file):
    all_rows = []
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".csv"):
            filepath = os.path.join(input_dir, filename)
            print(f"Processing file: {filepath}")
            rows = process_csv_file(filepath)
            all_rows.extend(rows)

    if all_rows:
        fieldnames = ["push", "board", "category", "title"]
        for key in all_rows[0].keys():
            if key not in fieldnames:
                fieldnames.append(key)

        with open(output_file, "w", newline="", encoding="utf-8") as out_f:
            writer = csv.DictWriter(
                out_f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL
            )
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Saved {len(all_rows)} cleaned rows to {output_file}")
    else:
        print("No data found to write.")


def main():
    input_dir = "data"
    output_file = "data-clean.csv"
    combine_csv_files(input_dir, output_file)


if __name__ == "__main__":
    main()
