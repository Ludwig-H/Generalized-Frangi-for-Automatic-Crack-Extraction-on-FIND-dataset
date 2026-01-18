import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from IPython.display import display, Markdown

# Configuration du style graphique
sns.set_theme(style="whitegrid", context="notebook", font_scale=1.1)

def display_noise_results(df):
    if df is None or df.empty:
        print("Le DataFrame est vide ou non défini.")
        return

    # ---------------------------------------------------------
    # 1. TABLEAU SYNTHÉTIQUE (MOYENNE ± STD)
    # ---------------------------------------------------------
    display(Markdown("### 📊 Synthèse Numérique (Moyenne ± Écart-type)"))

    # Sélection des colonnes métriques
    metrics_map = {
        "Jaccard": "Jaccard",
        "Tversky": "Tversky",
        "Wasserstein": "Wasserstein",
        "CSD_Jaccard": "CSD Jac.",
        "CSD_Tversky": "CSD Tvs.",
        "CSD_Wasserstein": "CSD Wass."
    }
    
    # On ne garde que les colonnes qui existent dans le df
    avail_cols = [c for c in metrics_map.keys() if c in df.columns]
    
    # Agrégation
    grouped = df.groupby(['NoiseExp', 'NoiseLevel'])[avail_cols].agg(['mean', 'std'])
    
    # Formatage propre "Moy ± Std"
    summary_df = pd.DataFrame(index=grouped.index)
    for col in avail_cols:
        short_name = metrics_map[col]
        mean_col = grouped[col]['mean']
        std_col = grouped[col]['std']
        # On gère les NaN pour CSD si non calculé
        summary_df[short_name] = mean_col.apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "-") + \
                                 " ± " + \
                                 std_col.apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "-")

    # Affichage du tableau stylisé
    display(summary_df.style.set_properties(**{'text-align': 'center'}).set_table_styles([
        dict(selector='th', props=[('text-align', 'center')])
    ]))

    # ---------------------------------------------------------
    # 2. VISUALISATION GRAPHIQUE
    # ---------------------------------------------------------
    display(Markdown("### 📈 Courbes de Robustesse"))

    experiments = df['NoiseExp'].unique()
    
    # Dictionnaire pour mapper les noms techniques vers des titres lisibles
    exp_titles = {
        "speckle_intensity": "Bruit Speckle (Intensité)",
        "gauss_range": "Bruit Gaussien (Profondeur)",
        "both": "Bruit Simultané (Intensité + Profondeur)"
    }

    # Métriques à tracer (Nom Colonne Ours, Nom Colonne CSD, Titre Axe Y)
    metrics_to_plot = [
        ("Jaccard", "CSD_Jaccard", "Index de Jaccard (Higher is better)"),
        ("Tversky", "CSD_Tversky", "Index de Tversky (Higher is better)"),
        ("Wasserstein", "CSD_Wasserstein", "Distance Wasserstein (Lower is better)")
    ]

    for exp in experiments:
        subset = df[df['NoiseExp'] == exp]
        if subset.empty: continue

        fig, axes = plt.subplots(1, 3, figsize=(20, 5), constrained_layout=True)
        fig.suptitle(f"Robustesse : {exp_titles.get(exp, exp)}", fontsize=16, weight='bold')

        for i, (our_col, csd_col, ylabel) in enumerate(metrics_to_plot):
            ax = axes[i]
            
            # Tracer "Ours" (Rouge)
            sns.lineplot(
                data=subset, x="NoiseLevel", y=our_col, 
                ax=ax, label="Ours (Frangi)", 
                color="#d62728", marker="o", linewidth=2.5, errorbar='sd'
            )

            # Tracer "CSD" (Vert) si disponible
            if csd_col in subset.columns and subset[csd_col].notna().any():
                sns.lineplot(
                    data=subset, x="NoiseLevel", y=csd_col, 
                    ax=ax, label="CrackSegDiff", 
                    color="#2ca02c", marker="s", linestyle="--", linewidth=2, errorbar='sd'
                )

            ax.set_title(our_col, fontsize=14)
            ax.set_xlabel("Niveau de Bruit (Variance / Sigma)", fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.legend(loc="best", frameon=True)
            
            # Inverser l'axe Y pour Wasserstein uniquement pour que "mieux" soit toujours "haut" ? 
            # Non, gardons la convention mathématique standard, mais on ajoute une grid.
            ax.grid(True, which='both', linestyle='--', alpha=0.7)

        plt.show()

# Exécution
if 'df_noise' in locals():
    display_noise_results(df_noise)
else:
    print("La variable 'df_noise' n'est pas définie. Assurez-vous d'avoir exécuté le benchmark auparavant.")
