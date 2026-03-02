"""
Projet ARS Occitanie – Cartographie des signaux (EIG/Réclamations) – Version ARS-friendly

Outputs:
- outputs/indicateurs_par_departement.csv
- outputs/note_bilan.md
- outputs/carte_taux_100k_occitanie.png
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt


# -------------------------
# Config
# -------------------------
OCC_DEPS = ["09", "11", "12", "30", "31", "32", "34", "46", "48", "65", "66", "81", "82"]
POP_PATH = "data_processed/donnees_departements.csv"
OUT_DIR = "outputs"
N_SIGNALS = 4000
DATE_START = "2024-01-01"
DATE_END = "2025-12-31"
SEED = 42


# -------------------------
# Helpers
# -------------------------
def ensure_out_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_minmax(s: pd.Series) -> pd.Series:
    return (s - s.min()) / (s.max() - s.min() + 1e-6)


def load_population_occitanie(pop_path: str) -> pd.DataFrame:
    """
    Expects columns like: REG, DEP, PTOT in donnees_departements.csv
    REG = 76 for Occitanie
    DEP can be numeric or string; we standardize to 2 chars with zfill (09, 31, ...)
    PTOT = population totale
    """
    pop_raw = pd.read_csv(
        pop_path,
        sep=None,          # auto-detect delimiter (tab/;/,)
        engine="python",
        dtype={"REG": "string", "DEP": "string"},
    )

    # Keep Occitanie rows
    pop_occ = pop_raw[pop_raw["REG"] == "76"].copy()
    pop_occ["dep"] = pop_occ["DEP"].str.strip().str.zfill(2)

    pop_occ["population"] = pd.to_numeric(pop_occ["PTOT"], errors="coerce")
    pop = pop_occ[["dep", "population"]].dropna()

    # Keep only Occitanie deps we care about
    pop = pop[pop["dep"].isin(OCC_DEPS)].copy()

    if pop.empty:
        raise ValueError("Population Occitanie introuvable (REG=76). Vérifie POP_PATH et colonnes REG/DEP/PTOT.")
    return pop


# -------------------------
# 1) Generate simulated signals (with motifs)
# -------------------------
def generate_signals(n: int) -> pd.DataFrame:
    np.random.seed(SEED)

    thematiques = [
        "Médicament",
        "Infection",
        "Prise en charge",
        "Identitovigilance",
        "Chirurgie",
        "Imagerie",
        "Urgences",
        "Maltraitance",
    ]

    # Motifs (plus “granulaires” que thématique) – utile pour démontrer l’analyse qualitative/quantitative
    motifs_by_theme = {
        "Médicament": ["Erreur de dosage", "Interaction médicamenteuse", "Retard d’administration", "Allergie non détectée"],
        "Infection": ["IAS postopératoire", "BMR", "Hygiène des mains", "Protocole non respecté"],
        "Prise en charge": ["Retard de diagnostic", "Coordination insuffisante", "Information patient", "Douleur non évaluée"],
        "Identitovigilance": ["Erreur d’identité", "Dossier patient", "Étiquetage prélèvement", "Homonymie"],
        "Chirurgie": ["Check-list bloc", "Complication per-op", "Dispositif médical", "Traçabilité"],
        "Imagerie": ["Injection produit", "Erreur d’examen", "Délai de rendu", "Radioprotection"],
        "Urgences": ["Temps d’attente", "Orientation", "Risque suicidaire", "Sortie non sécurisée"],
        "Maltraitance": ["Comportement inadapté", "Respect de la dignité", "Non-consentement", "Vulnérabilité"],
    }

    types = ["Réclamation", "EIG", "Signalement"]
    statuts = ["Nouveau", "En cours", "Clôturé"]

    dates = pd.date_range(DATE_START, DATE_END, freq="D")

    df = pd.DataFrame({
        "date_signal": np.random.choice(dates, n),
        "dep": np.random.choice(OCC_DEPS, n),
        "type_signal": np.random.choice(types, n, p=[0.65, 0.20, 0.15]),
        "thematique": np.random.choice(thematiques, n),
        "gravite": np.random.choice([1, 2, 3, 4], n, p=[0.45, 0.35, 0.15, 0.05]),
        "statut": np.random.choice(statuts, n, p=[0.15, 0.35, 0.50]),
    })

    # Build motif from thematique
    df["motif"] = df["thematique"].map(lambda t: np.random.choice(motifs_by_theme[t]))

    # Delay correlated with gravity (simple, explainable)
    base = np.random.gamma(shape=3, scale=10, size=n)
    df["delai_traitement_j"] = (base + (df["gravite"] - 1) * 8).round().astype(int)

    return df


# -------------------------
# 2) Indicators + risk score + taux/100k + quarterly evolution
# -------------------------
def build_indicators(df: pd.DataFrame, pop: pd.DataFrame) -> pd.DataFrame:
    # Base aggregation per dep
    agg = df.groupby("dep").agg(
        nb_signaux=("dep", "size"),
        gravite_moy=("gravite", "mean"),
        delai_med=("delai_traitement_j", "median"),
        part_eig=("type_signal", lambda s: (s == "EIG").mean() * 100),
        part_grave_34=("gravite", lambda s: (s >= 3).mean() * 100),
    ).reset_index()

    # Merge population + taux
    agg = agg.merge(pop, on="dep", how="left")
    agg["taux_100k"] = (agg["nb_signaux"] / agg["population"]) * 100_000

    # Quarterly evolution (last quarter vs previous)
    df2 = df.copy()
    df2["trimestre"] = df2["date_signal"].dt.to_period("Q")
    vol_trim = df2.groupby(["dep", "trimestre"]).size().unstack(fill_value=0)

    last_q = df2["trimestre"].max()
    prev_q = last_q - 1

    if last_q not in vol_trim.columns:
        vol_trim[last_q] = 0
    if prev_q not in vol_trim.columns:
        vol_trim[prev_q] = 0

    evolution_pct = ((vol_trim[last_q] - vol_trim[prev_q]) / vol_trim[prev_q].replace(0, 1)) * 100
    evolution_pct = evolution_pct.rename("evolution_pct").reset_index()

    agg = agg.merge(evolution_pct, on="dep", how="left").fillna({"evolution_pct": 0})

    # Risk score 0–100 (simple, explainable)
    # Here: frequency (taux), severity, delays, and recent evolution.
    agg["taux_norm"] = normalize_minmax(agg["taux_100k"])
    agg["grav_norm"] = normalize_minmax(agg["gravite_moy"])
    agg["delai_norm"] = normalize_minmax(agg["delai_med"])
    agg["evo_norm"] = normalize_minmax(agg["evolution_pct"])

    agg["score_risque"] = (
        0.40 * agg["taux_norm"] +
        0.30 * agg["grav_norm"] +
        0.20 * agg["evo_norm"] +
        0.10 * agg["delai_norm"]
    ) * 100

    # Sort for readability
    agg = agg.sort_values("score_risque", ascending=False).reset_index(drop=True)
    return agg


# -------------------------
# 3) Map PNG (Option B: taux/100k)
# -------------------------
def plot_map_taux_png(agg: pd.DataFrame, out_path: str) -> None:
    geo_url = "https://etalab-datasets.geo.data.gouv.fr/contours-administratifs/latest/geojson/departements-1000m.geojson"
    deps = gpd.read_file(geo_url)

    deps_occ = deps[deps["code"].isin(OCC_DEPS)].copy()
    gdf = deps_occ.merge(agg, left_on="code", right_on="dep", how="left")

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    gdf.plot(
        column="taux_100k",
        linewidth=0.8,
        edgecolor="black",
        legend=True,
        ax=ax,
    )

    ax.set_title("Taux de signaux / 100 000 habitants\nOccitanie – 2024-2025", fontsize=14)
    ax.axis("off")

    # Label each dep with taux (lightweight labels)
    for _, row in gdf.iterrows():
        if row.geometry is None or pd.isna(row["taux_100k"]):
            continue
        x, y = row.geometry.representative_point().coords[0]
        ax.text(x, y, f"{row['taux_100k']:.1f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


# -------------------------
# 4) One-page note (Markdown)
# -------------------------
def write_one_pager(df: pd.DataFrame, agg: pd.DataFrame, pop: pd.DataFrame, out_path: str) -> None:
    periode = f"{df['date_signal'].min().date()} → {df['date_signal'].max().date()}"

    total = len(df)
    part_eig = (df["type_signal"].eq("EIG").mean() * 100)
    part_grave = (df["gravite"].ge(3).mean() * 100)
    delai_med_reg = df["delai_traitement_j"].median()

    pop_total = pop["population"].sum()
    taux_regional = (total / pop_total) * 100_000

    top_score = agg.head(3)[["dep", "score_risque", "taux_100k", "evolution_pct"]]
    top_taux = agg.sort_values("taux_100k", ascending=False).head(3)[["dep", "taux_100k"]]
    top_evo = agg.sort_values("evolution_pct", ascending=False).head(3)[["dep", "evolution_pct"]]

    top_theme = df["thematique"].value_counts().head(3)
    top_motif = df["motif"].value_counts().head(3)

    note = f"""# Bilan – Signaux qualité & sécurité (Occitanie)

**Période analysée :** {periode}

## Chiffres clés
- **Volume total :** {total}
- **Part EIG :** {part_eig:.1f} %
- **Part gravité 3–4 :** {part_grave:.1f} %
- **Délai médian de traitement :** {delai_med_reg:.0f} jours
- **Taux régional :** {taux_regional:.1f} signaux / 100 000 hab

## Départements prioritaires (score de risque)
{top_score.to_string(index=False)}

## Départements avec taux le plus élevé (/100k)
{top_taux.to_string(index=False)}

## Plus fortes évolutions trimestrielles (%)
{top_evo.to_string(index=False)}

## Top thématiques
{top_theme.to_string()}

## Top motifs
{top_motif.to_string()}
"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(note)


# -------------------------
# Main
# -------------------------
def main() -> None:
    ensure_out_dir(OUT_DIR)

    # 1) Data
    df = generate_signals(N_SIGNALS)

    # 2) Population (Occitanie)
    pop = load_population_occitanie(POP_PATH)

    # 3) Indicators
    agg = build_indicators(df, pop)

    # 4) Exports
    agg.to_csv(f"{OUT_DIR}/indicateurs_par_departement.csv", index=False)
    print(f"CSV généré: {OUT_DIR}/indicateurs_par_departement.csv")

    # 5) Map (Option B)
    plot_map_taux_png(agg, f"{OUT_DIR}/carte_taux_100k_occitanie.png")
    print(f"Carte générée: {OUT_DIR}/carte_taux_100k_occitanie.png")

    # 6) One-pager
    write_one_pager(df, agg, pop, f"{OUT_DIR}/note_bilan.md")
    print(f"Note générée: {OUT_DIR}/note_bilan.md")

    # Quick console summary
    print("\nTop 5 – score de risque :")
    print(agg[["dep", "score_risque", "taux_100k", "evolution_pct"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()