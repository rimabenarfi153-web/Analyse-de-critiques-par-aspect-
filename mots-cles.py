import re
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

ASPECTS = ["scenario", "acteurs", "realisation", "musique", "duree"]

MOTS = {
    "scenario": ["scénario", "scenario", "histoire", "intrigue", "dialogue", "récit"],
    "acteurs": ["acteur", "actrice", "jeu", "interprétation", "casting", "comédien"],
    "realisation": ["réalisation", "réalisateur", "mise en scène", "image",
                    "effets spéciaux", "photographie"],
    "musique": ["musique", "bande originale", "bande-son"],
    "duree": ["trop long", "trop court", "durée", "longueur", "longuet", "interminable"],
}

POS = ["bon", "bien", "excellent", "magnifique", "superbe", "génial", "super",
       "réussi", "original", "beau", "belle", "top", "parfait", "bravo",
       "adoré", "aimé", "formidable", "touchant", "remarquable"]
NEG = ["mauvais", "nul", "nulle", "ennuyeux", "décevant", "déçu", "raté",
       "faible", "moche", "lent", "mou", "molle", "pauvre", "médiocre",
       "horrible", "catastrophe", "bof", "plat", "long", "pire"]


def contient(texte, mots):
    return any(re.search(r"\b" + re.escape(m), texte) for m in mots)


def predire(critique):
    texte = str(critique).lower()
    phrases = re.split(r"[.!?;\n]", texte)
    res = {}
    for asp in ASPECTS:
        sel = [p for p in phrases if contient(p, MOTS[asp])]
        if not sel:
            continue
        pos = sum(contient(p, POS) for p in sel)
        neg = sum(contient(p, NEG) for p in sel)
        res[asp] = "positif" if pos > neg else "negatif" if neg > pos else "neutre"
    return res


df = pd.read_excel("annotation.xlsx")
preds = [predire(c) for c in df["critique"]]

lignes = []
for asp in ASPECTS:
    gold, pred = [], []
    bons, commun = 0, 0
    for (_, row), p in zip(df.iterrows(), preds):
        g = row[asp]
        g_present = isinstance(g, str) and g.strip() != ""
        gold.append(int(g_present))
        pred.append(int(asp in p))
        if g_present and asp in p:
            commun += 1
            if g.strip() == p[asp]:
                bons += 1
    lignes.append({
        "aspect": asp,
        "precision": round(precision_score(gold, pred, zero_division=0), 2),
        "rappel": round(recall_score(gold, pred, zero_division=0), 2),
        "F1": round(f1_score(gold, pred, zero_division=0), 2),
        "sentiment correct": f"{bons}/{commun}",
    })

res = pd.DataFrame(lignes)
print(res.to_string(index=False))
res.to_csv("resultats_mots_cles.csv", index=False, encoding="utf-8-sig")