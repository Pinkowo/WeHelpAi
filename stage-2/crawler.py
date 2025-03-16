import requests
from bs4 import BeautifulSoup
import csv
import time

max_articles = 200000


def parse_push_count(nrec_div):
    if nrec_div is None:
        return 0
    text = nrec_div.get_text().strip()
    if text == "":
        return 0
    elif text == "爆":
        return 99
    else:
        try:
            return int(text)
        except ValueError:
            return 0


def crawl_ptt(board, max_articles=200000, delay=0.5):
    base_url = "https://www.ptt.cc"
    current_url = f"{base_url}/bbs/{board}/index.html"
    articles = []

    while current_url and len(articles) < max_articles:
        response = requests.get(current_url, cookies={"over18": "1"})
        if response.status_code != 200:
            print("Failed to retrieve:", current_url)
            break

        soup = BeautifulSoup(response.text, "html.parser")
        entries = soup.find_all("div", class_="r-ent")

        for entry in entries:
            nrec_div = entry.find("div", class_="nrec")
            push = parse_push_count(nrec_div)

            title_div = entry.find("div", class_="title")
            title_a = title_div.a if title_div else None
            title = title_a.text.strip() if title_a else "deleted"

            articles.append(
                {
                    "push": push,
                    "board": board,
                    "title": title,
                }
            )

            if len(articles) >= max_articles:
                break

        btn_group = soup.select("div.btn-group-paging a")
        if btn_group and len(btn_group) >= 2:
            prev_href = btn_group[1].get("href")
            current_url = base_url + prev_href
        else:
            break

        print(f"Crawled {len(articles)} articles...")
        time.sleep(delay)

    return articles


def save_to_csv(articles, filename="ptt_articles.csv"):
    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["push", "board", "title"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for article in articles:
            writer.writerow(article)
    print(f"Saved {len(articles)} articles to {filename}")


def main():
    board_list = [
        "baseball",
        "Boy-Girl",
        "c_chat",
        "hatepolitics",
        "Lifeismoney",
        "Military",
        "pc_shopping",
        "stock",
        "Tech_Job",
    ]

    for board in board_list:
        articles = crawl_ptt(board, max_articles)
        save_to_csv(articles, filename=f"data/{board}.csv")


if __name__ == "__main__":
    main()
