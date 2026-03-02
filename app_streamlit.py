import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
from streamlit_folium import st_folium
import folium

st.set_page_config(page_title="Dashboard Signaux ARS Occitanie", layout="wide")

st.title("Dashboard interactif – Qualité & sécurité (Occitanie)")

# --- Load data ---
@st.cache_data
def load_data(csv_path="outputs/indicateurs_par_departement.csv"):
    df = pd.read_csv(csv_path, dtype={"dep": "string"})
    return df

df = load_data()

# --- Sidebar controls ---
st.sidebar.header("Filtres & options")
metric = st.sidebar.selectbox(
    "Indicateur à cartographier",
    options=["taux_100k", "score_risque", "evolution_pct"],
    format_func=lambda x: {
        "taux_100k": "Taux / 100 000 hab",
        "score_risque": "Score de risque (0–100)",
        "evolution_pct": "Évolution trimestrielle (%)",
    }[x]
)

top_n = st.sidebar.slider("Top N départements", 3, 13, 5)

st.sidebar.markdown("---")
uploaded = st.sidebar.file_uploader("📁 Uploader un CSV indicateurs", type="csv")
if uploaded is not None:
    df = load_data(uploaded)

# --- KPIs ---
st.markdown("## Indicateurs clés – Région Occitanie")
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total signaux", df["nb_signaux"].sum())
col2.metric("Taux régional (/100k)", f"{(df['nb_signaux'].sum() / df['population'].sum() * 100000):.1f}")
col3.metric("Délai médian (j)", f"{df['delai_med'].median():.0f}")
col4.metric("Part EIG (%)", f"{df['part_eig'].mean():.1f}")

st.markdown("---")

# --- Interprétation des KPI (automatique) ---

total_signaux = int(df["nb_signaux"].sum())
taux_reg = (df["nb_signaux"].sum() / df["population"].sum() * 100000)
delai_med = float(df["delai_med"].median())
part_eig = float(df["part_eig"].mean())

top_taux = df.sort_values("taux_100k", ascending=False).head(3)[["dep","taux_100k"]]
top_score = df.sort_values("score_risque", ascending=False).head(3)[["dep","score_risque"]]
top_evo = df.sort_values("evolution_pct", ascending=False).head(3)[["dep","evolution_pct"]]

st.write(
    f"""
**Synthèse régionale (Occitanie)**  
- **{total_signaux} signaux** sur la période analysée.  
- **Taux régional estimé : {taux_reg:.1f} / 100 000 habitants**, permettant une comparaison territoriale standardisée.  
- **Délai médian : {delai_med:.0f} jours**, indicateur de performance de traitement (à suivre dans le temps).  
- **Part moyenne d’EIG : {part_eig:.1f}%**, proportion de signaux à criticité potentiellement élevée.
"""
)

st.markdown("Top 3")
c1, c2, c3 = st.columns(3)

with c1:
    st.caption("Départements – **taux** le plus élevé (/100k)")
    st.dataframe(top_taux, use_container_width=True)

with c2:
    st.caption("Départements – **score de risque** le plus élevé (0–100)")
    st.dataframe(top_score, use_container_width=True)

with c3:
    st.caption("Départements – **hausse trimestrielle** la plus forte (%)")
    st.dataframe(top_evo, use_container_width=True)

st.markdown("### À retenir ")
st.info("""
• Prioriser l’analyse des territoires présentant un taux élevé et/ou une hausse trimestrielle marquée.  
• Croiser ces résultats avec les thématiques/motifs pour cibler des plans de contrôle.  
• Suivre le délai médian et la part d’EIG comme indicateurs de pilotage (tendance trimestrielle).
""")

# --- Carte interactive (Folium) ---
st.markdown("### Carte interactive")

# Load geojson
geo_url = "https://etalab-datasets.geo.data.gouv.fr/contours-administratifs/latest/geojson/departements-1000m.geojson"
deps = gpd.read_file(geo_url)
OCC_DEPS = ["09","11","12","30","31","32","34","46","48","65","66","81","82"]
deps_occ = deps[deps["code"].isin(OCC_DEPS)].copy()

gdf = deps_occ.merge(df, left_on="code", right_on="dep", how="left")

m = folium.Map(location=[43.7, 1.6], zoom_start=7)

folium.Choropleth(
    geo_data=gdf,
    data=gdf,
    columns=["code", metric],
    key_on="feature.properties.code",
    fill_color="YlOrRd",
    fill_opacity=0.7,
    line_opacity=0.5,
    legend_name=str(metric)
).add_to(m)

for _, r in gdf.iterrows():
    if pd.notna(r[metric]):
        folium.CircleMarker(
            location=[r.geometry.centroid.y, r.geometry.centroid.x],
            radius=6,
            color="blue",
            fill=True,
            fill_opacity=0.7,
            popup=(
                f"<b>{r['nom']} ({r['code']})</b><br>"
                f"{metric}: {r[metric]:.1f}<br>"
                f"Nb signaux: {r['nb_signaux']}<br>"
                f"Délai méd: {r['delai_med']} j"
            )
        ).add_to(m)

st_folium(m, width=800)

st.markdown("---")

# --- Classements ---
st.markdown("Classements par départements")

st.subheader("Top par score de risque")
st.dataframe(df.sort_values("score_risque", ascending=False).head(top_n))

st.subheader("Top par taux / 100 k hab")
st.dataframe(df.sort_values("taux_100k", ascending=False).head(top_n))

st.subheader("Top par évolution trimestrielle (%)")
st.dataframe(df.sort_values("evolution_pct", ascending=False).head(top_n))

st.markdown("---")

# --- Graphiques ---
fig1 = px.bar(
    df.sort_values(metric, ascending=False).head(top_n),
    x="dep",
    y=metric,
    title=f"Top {top_n} – {metric}",
    labels={"dep":"Département", metric: metric}
)
st.plotly_chart(fig1, use_container_width=True)

fig2 = px.scatter(
    df, x="taux_100k", y="score_risque", text="dep",
    labels={"taux_100k":"Taux /100k","score_risque":"Score de risque"},
    title="Relation Taux /100k vs Score de risque"
)
fig2.update_traces(textposition="top center")
st.plotly_chart(fig2, use_container_width=True)

with st.expander("ℹ️ Méthodologie (prototype)"):
    st.write(
        """
- Données de signaux simulées pour démontrer la méthodologie (transposable aux SI métiers type REC-SIRENA / ICEA).
- Indicateurs : volumétrie, gravité, délais, évolution trimestrielle, normalisation population (taux /100k).
- Score de risque (0–100) : combinaison pondérée taux, gravité, évolution et délais (approche explicable).
"""
    )
