import json
import time
import pandas as pd
import ollama
from pydantic import BaseModel
from typing import Literal, List


class Aspect(BaseModel):
    aspect: Literal["scenario", "acteurs", "realisation", "musique", "duree"]
    sentiment: Literal["positif", "negatif", "neutre"]
    extrait: str


class Resultat(BaseModel):
    aspects: List[Aspect]


PROMPT = (
    "Analyse cette critique de film. Repère les aspects dont elle parle "
    "parmi : scenario, acteurs, realisation, musique, duree. "
    "Pour chaque aspect, donne le sentiment (positif, negatif ou neutre) "
    "et l'extrait de la critique. Si un aspect n'est pas mentionné, "
    "ne l'ajoute pas.\n\nCritique : "
)

df = pd.read_excel("annotation.xlsx")
resultats = {}

for n, row in df.iterrows():
    i = int(row["id"])
    debut = time.time()
    try:
        r = ollama.chat(
            model="qwen2.5:1.5b",
            messages=[{"role": "user", "content": PROMPT + str(row["critique"])}],
            format=Resultat.model_json_schema(),
        )
        res = Resultat.model_validate_json(r["message"]["content"])
        resultats[i] = {"valide": True, "aspects": [a.model_dump() for a in res.aspects]}
    except Exception as e:
        resultats[i] = {"valide": False, "erreur": str(e)}
    with open("predictions.json", "w", encoding="utf-8") as f:
        json.dump(resultats, f, ensure_ascii=False, indent=2)
    print(n + 1, "/", len(df), "-", round(time.time() - debut), "s", flush=True)

print("Fini.")