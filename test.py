from datasets import load_dataset
import ollama

ds = load_dataset(
    "tblard/allocine",
    data_files={"test": "allocine/test-00000-of-00001.parquet"},
    split="test",
)
critique = ds[0]["review"]
print("CRITIQUE:", critique[:300])

r = ollama.chat(
    model="qwen2.5:1.5b",
    messages=[{
        "role": "user",
        "content": "Résume en une phrase cette critique de film : " + critique,
    }],
)
print("RESUME:", r["message"]["content"])