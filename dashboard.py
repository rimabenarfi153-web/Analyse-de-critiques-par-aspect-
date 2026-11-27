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
    with open("predictions_300.json", encoding="utf-8") as f:
        preds300 = json.load(f)
    return df, preds, r_llm, r_mc, preds300


df, preds, r_llm, r_mc, preds300 = charger()

tab1, tab2, tab3, tab4 = st.tabs(
    ["Comparaison", "Critique par critique", "Aspects du modèle",
     "Vue 300 critiques"]
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

with tab4:
    st.subheader("Vue d'ensemble sur 300 critiques")
    st.warning(
        "Attention : le modèle invente beaucoup d'aspects (surtout musique "
        "et durée). Cette distribution est donc gonflée. Seules les 50 "
        "critiques annotées à la main servent de référence."
    )

    valides = [p for p in preds300.values() if p["valide"]]
    st.write("Critiques valides :", len(valides), "/", len(preds300))

    rows = []
    for p in valides:
        for a in p["aspects"]:
            rows.append({"aspect": a["aspect"], "sentiment": a["sentiment"]})
    d300 = pd.DataFrame(rows)
    table300 = pd.crosstab(d300["aspect"], d300["sentiment"])
    st.subheader("Aspects et sentiments (300 critiques)")
    st.bar_chart(table300)
    st.dataframe(table300)

    nb_mod = {a: 0 for a in ASPECTS}
    for p in valides:
        for asp in {x["aspect"] for x in p["aspects"]}:
            nb_mod[asp] += 1
    nb_man = {}
    for asp in ASPECTS:
        col = df[asp].fillna("").astype(str).str.strip()
        nb_man[asp] = int((col != "").sum())
    taux = pd.DataFrame({
        "Modèle (sur 300)": {a: 100 * nb_mod[a] / len(valides) for a in ASPECTS},
        "Annotation manuelle (sur 50)": {a: 100 * nb_man[a] / len(df) for a in ASPECTS},
    })
    st.subheader("Taux de mention par aspect (%)")
    st.bar_chart(taux)
    st.dataframe(taux.round(1))