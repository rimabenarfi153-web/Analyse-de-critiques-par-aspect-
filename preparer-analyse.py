import pandas as pd

err = pd.read_csv("erreurs.csv", encoding="utf-8-sig")
ann = pd.read_excel("annotation.xlsx")[["id", "critique"]]
ann = ann.rename(columns={"critique": "critique_complete"})

quotas = {"aspect invente": 4, "mauvais sentiment": 4, "aspect rate": 2}

parts = []
for t, n in quotas.items():
    sel = err[err["type"] == t]
    parts.append(sel.sample(min(n, len(sel)), random_state=42))

ech = pd.concat(parts)
ech = ech.merge(ann, on="id", how="left")
ech = ech[["id", "aspect", "type", "toi", "modele", "critique_complete"]]

# خانات فارغة تعمرها إنت
ech["modele_a_raison"] = ""
ech["cause"] = ""
ech["commentaire"] = ""

ech.to_excel("analyse_erreurs.xlsx", index=False)
print(ech[["id", "aspect", "type"]])