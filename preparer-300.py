from datasets import load_dataset
import pandas as pd

ds = load_dataset("tblard/allocine",
    data_files={"test": "allocine/test-00000-of-00001.parquet"}, split="test")
tout = ds.to_pandas()

ann = pd.read_excel("annotation.xlsx")
ids50 = [int(x) for x in ann["id"]]

# تأكد أن id = رقم السطر في الداتا
ok = 0
for i, txt in zip(ids50, ann["critique"]):
    if str(tout.loc[i, "review"]).strip() == str(txt).strip():
        ok += 1
print("ids qui correspondent:", ok, "/ 50")

# نفس شرط الطول متاع الـ 50
ok_len = tout[tout["review"].str.len().between(150, 500)]
reste = ok_len.drop(index=ids50, errors="ignore")
autres = reste.sample(250, random_state=42)

df300 = pd.concat([tout.loc[ids50], autres])
df300 = pd.DataFrame({
    "id": df300.index,
    "critique": df300["review"].values,
    "annote": [i in ids50 for i in df300.index],
})
df300.to_excel("critiques_300.xlsx", index=False)
print("Total:", len(df300))