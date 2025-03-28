import csv
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

folder = "test-data"  # "train-data" or "test-data"


def main():
    csv_file = f"{folder}/data-clean-token.csv"

    print("Titles Ready")

    test_documents = []
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if idx >= 1000:
                break
            tokens = row["title"].split()
            test_documents.append(TaggedDocument(words=tokens, tags=[str(idx)]))

    print("Tagged Documents Ready")

    print("Start Training")
    model = Doc2Vec.load("doc2vec_model/doc2vec_model.bin")

    print("Test Similarity")

    correct_self_top1 = 0
    correct_self_top2 = 0
    total_count = len(test_documents)

    for i, doc in enumerate(test_documents):
        if i % 100 == 0:
            print(i)

        tokens = doc.words
        inferred_vector = model.infer_vector(tokens, epochs=500)

        top2 = model.dv.most_similar([inferred_vector], topn=2)

        if top2[0][0] == str(i):
            correct_self_top1 += 1

        if top2[0][0] == str(i) or top2[1][0] == str(i):
            correct_self_top2 += 1

    print(total_count)

    self_similarity = correct_self_top1 / total_count
    second_self_similarity = correct_self_top2 / total_count

    print(f"Self Similarity {self_similarity:.3f}")
    print(f"Second Self Similarity {second_self_similarity:.3f}")


if __name__ == "__main__":
    main()
