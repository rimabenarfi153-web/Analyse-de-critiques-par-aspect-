import pandas as pd

df = pd.read_excel("annotation.xlsx")
aspects = ["scenario", "acteurs", "realisation", "musique", "duree"]

for a in aspects:
    print("-----", a)
    print(df[a].value_counts(dropna=False))