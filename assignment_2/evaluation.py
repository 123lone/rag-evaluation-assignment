import json
import time
from assignment_1.retriever import Retriever

def evaluate():
    data = json.load(open("assignment_2/test_dataset.json"))
    retriever = Retriever()
    results = []

    for item in data["test_questions"]:
        q = item["question"]
        gold = set(item["source_documents"])

        start = time.time()
        res = retriever.retrieve(q, k=5)
        latency = time.time() - start

        retrieved = [rid.split("-")[0] for rid in res["ids"][0]]
        retrieved_set = set(retrieved)

        precision = len(retrieved_set & gold) / len(retrieved) if retrieved else 0
        recall = len(retrieved_set & gold) / len(gold) if gold else 0
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        results.append({
            "query": q,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "latency": latency
        })

    json.dump(results, open("assignment_2/test_results.json", "w"), indent=4)
    print("Evaluation complete.")

if __name__ == "__main__":
    evaluate()
