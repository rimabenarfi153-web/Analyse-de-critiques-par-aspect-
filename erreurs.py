import json
import pandas as pd

ASPECTS = ["scenario", "acteurs", "realisation", "musique", "duree"]

df = pd.read_excel("annotation.xlsx")
with open("predictions.json", encoding="utf-8") as f:
    preds = json.load(f)

lignes = []
for _, row in df.iterrows():
    p = preds.get(str(int(row["id"])))
    pred_asp = {}
    if p and p["valide"]:
        pred_asp = {a["aspect"]: a["sentiment"] for a in p["aspects"]}
    for asp in ASPECTS:
        g = row[asp]
        g = g.strip() if isinstance(g, str) else ""
        m = pred_asp.get(asp, "")
        if g and not m:
            type_err = "aspect rate"
        elif m and not g:
            type_err = "aspect invente"
        elif g and m and g != m:
            type_err = "mauvais sentiment"
        else:
            continue
        lignes.append({
            "id": int(row["id"]),
            "aspect": asp,
            "type": type_err,
            "toi": g,
            "modele": m,
            "critique": str(row["critique"])[:200],
        })

err = pd.DataFrame(lignes)
err.to_csv("erreurs.csv", index=False, encoding="utf-8-sig")
print(err["type"].value_counts())
print()
print(pd.crosstab(err["aspect"], err["type"]))