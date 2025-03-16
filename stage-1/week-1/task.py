import urllib.request
import urllib.parse
import json
import time
import statistics
import csv


def scrape_data_api():
    base_url = "https://ecshweb.pchome.com.tw/search/v4.3/all/results"

    page = 1
    page_count = 10
    all_items = []
    i5_prices = []

    while True:
        query_params = {
            "cateid": "DSAA31",
            "attr": "",
            "pageCount": str(page_count),
            "page": str(page),
        }

        query_string = urllib.parse.urlencode(query_params)
        full_url = f"{base_url}?{query_string}"

        with urllib.request.urlopen(full_url) as response:
            data_str = response.read().decode("utf-8", errors="ignore")

        data_obj = json.loads(data_str)

        prods = data_obj.get("Prods", [])
        if not prods:
            i5_ave_price = statistics.mean(i5_prices)
            print(f"使用 i5 處理器的產品平均價格是 {i5_ave_price}")
            break

        for item in prods:
            product_id = item.get("Id", "")
            product_name = item.get("Name", "")
            product_price = item.get("Price", "")
            product_rating = item.get("ratingValue", 0.0)
            product_review_count = item.get("reviewCount", 0)

            all_items.append(
                {
                    "id": product_id,
                    "name": product_name,
                    "price": product_price,
                    "rating": product_rating,
                    "review_count": product_review_count,
                }
            )

            if "i5" in product_name:
                i5_prices.append(product_price)

        time.sleep(0.5)

        page += 1
    return all_items


def is_best_products(prod):
    if prod["review_count"] is None:
        return False
    elif prod["rating"] is None:
        return False
    elif prod["review_count"] == 0:
        return False
    elif prod["rating"] <= 4.9:
        return False
    else:
        return True


def is_i5_processor(prod):
    if "i5" in prod["name"]:
        return True
    else:
        return False


def calc_z_score(price, pstdev, mean):
    return (price - mean) / pstdev


def write_to_csv(product_list):
    with open("standardization.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ProductID", "Price", "PriceZScore"])

        prices = [prod["price"] for prod in product_list]
        pstdev = statistics.pstdev(prices)
        mean = statistics.mean(prices)

        for prod in product_list:
            product_id = prod.get("id", "")
            price = prod.get("price", "")
            price_z_score = calc_z_score(price, pstdev, mean)

            writer.writerow([product_id, price, round(price_z_score, 2)])


if __name__ == "__main__":
    results = scrape_data_api()
    print(f"共抓取到 {len(results)} 筆商品資料")

    with open("products.txt", "w", encoding="utf-8") as f:
        for prod in results:
            f.write(prod["id"] + "\n")

    with open("best-products.txt", "w", encoding="utf-8") as f:
        best_products = filter(is_best_products, results)
        print(
            f"至少有 1 則評論且評分高於 4.9 的商品資料有 {len(list(best_products))} 筆"
        )
        for prod in best_products:
            f.write(prod["id"] + "\n")

    write_to_csv(results)
