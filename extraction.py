from datasets import load_dataset
from pydantic import BaseModel
from typing import Literal, List
import ollama


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

ds = load_dataset(
    "tblard/allocine",
    data_files={"test": "allocine/test-00000-of-00001.parquet"},
    split="test",
)

N = 2
valides = 0
for i in range(N):
    critique = ds[i]["review"]
    r = ollama.chat(
        model="qwen2.5:1.5b",
        messages=[{"role": "user", "content": PROMPT + critique}],
        format=Resultat.model_json_schema(),
    )
    try:
        res = Resultat.model_validate_json(r["message"]["content"])
        valides += 1
        print("---", i, critique[:80])
        for a in res.aspects:
            print("  ", a.aspect, a.sentiment, "|", a.extrait)
    except Exception as e:
        print("--- INVALIDE", i, e)

print("JSON valides :", valides, "/", N)