import json
import pandas as pd

with open("predictions_300.json", encoding="utf-8") as f:
    preds = json.load(f)

print("Total:", len(preds))
valides = [p for p in preds.values() if p["valide"]]
print("JSON valides:", len(valides), "/", len(preds))

# عدد الجوانب لكل نقد
nb = [len(p["aspects"]) for p in valides]
print("Aspects par critique (moyenne):", round(sum(nb) / len(nb), 2))

# توزيع الجوانب والأحاسيس
lignes = []
for p in valides:
    for a in p["aspects"]:
        lignes.append((a["aspect"], a["sentiment"]))
d = pd.DataFrame(lignes, columns=["aspect", "sentiment"])
print()
print(pd.crosstab(d["aspect"], d["sentiment"]))