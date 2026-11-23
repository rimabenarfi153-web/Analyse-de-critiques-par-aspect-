import json
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

ASPECTS = ["scenario", "acteurs", "realisation", "musique", "duree"]

df = pd.read_excel("annotation.xlsx")
with open("predictions.json", encoding="utf-8") as f:
    preds = json.load(f)

valides = sum(1 for p in preds.values() if p["valide"])
print("JSON valides :", valides, "/", len(preds))

lignes = []
for asp in ASPECTS:
    gold, pred = [], []
    bons, commun = 0, 0
    for _, row in df.iterrows():
        p = preds.get(str(int(row["id"])))
        if p is None or not p["valide"]:
            pred_asp = {}
        else:
            pred_asp = {a["aspect"]: a["sentiment"] for a in p["aspects"]}
        g = row[asp]
        g_present = isinstance(g, str) and g.strip() != ""
        gold.append(int(g_present))
        pred.append(int(asp in pred_asp))
        if g_present and asp in pred_asp:
            commun += 1
            if g.strip() == pred_asp[asp]:
                bons += 1
    lignes.append({
        "aspect": asp,
        "mentions (toi)": sum(gold),
        "precision": round(precision_score(gold, pred, zero_division=0), 2),
        "rappel": round(recall_score(gold, pred, zero_division=0), 2),
        "F1": round(f1_score(gold, pred, zero_division=0), 2),
        "sentiment correct": f"{bons}/{commun}",
    })

res = pd.DataFrame(lignes)
print(res.to_string(index=False))
res.to_csv("resultats.csv", index=False, encoding="utf-8-sig")