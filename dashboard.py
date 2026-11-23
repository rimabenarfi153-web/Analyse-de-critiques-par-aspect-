import json
import pandas as pd
import streamlit as st

ASPECTS = ["scenario", "acteurs", "realisation", "musique", "duree"]

st.set_page_config(page_title="Critiques de films", layout="wide")
st.title("Analyse des critiques de films par aspect")
@st.cache_data
def charger():
    df = pd.read_excel("annotation.xlsx")
    with open("predictions.json", encoding="utf-8") as f:
        preds = json.load(f)
    r_llm = pd.read_csv("resultats.csv")
    r_mc = pd.read_csv("resultats_mots_cles.csv")
    return df, preds, r_llm, r_mc


df, preds, r_llm, r_mc = charger()

tab1, tab2, tab3 = st.tabs(
    ["Comparaison", "Critique par critique", "Aspects du modèle"]
)

with tab1:
    st.subheader("Modèle (Ollama)")
    st.dataframe(r_llm, hide_index=True)
    st.subheader("Méthode par mots-clés")
    st.dataframe(r_mc, hide_index=True)
    comp = pd.DataFrame({
        "Modèle": r_llm.set_index("aspect")["F1"],
        "Mots-clés": r_mc.set_index("aspect")["F1"],
    })
    st.subheader("F1 par aspect")
    st.bar_chart(comp)

with tab2:
    choix = st.selectbox("Choisir une critique (id)", df["id"].tolist())
    row = df[df["id"] == choix].iloc[0]
    st.write(row["critique"])
    p = preds.get(str(int(choix)))
    pred_asp = {}
    if p and p["valide"]:
        pred_asp = {a["aspect"]: a["sentiment"] for a in p["aspects"]}
    lignes = []
    for asp in ASPECTS:
        g = row[asp]
        g = g if isinstance(g, str) and g.strip() else "-"
        lignes.append({"aspect": asp, "annotation (toi)": g,
                       "modèle": pred_asp.get(asp, "-")})
    st.table(pd.DataFrame(lignes))

with tab3:
    rows = []
    for p in preds.values():
        if p["valide"]:
            for a in p["aspects"]:
                rows.append({"aspect": a["aspect"], "sentiment": a["sentiment"]})
    d = pd.DataFrame(rows)
    table = pd.crosstab(d["aspect"], d["sentiment"])
    st.subheader("Aspects détectés par le modèle et leur sentiment")
    st.bar_chart(table)
    st.dataframe(table)