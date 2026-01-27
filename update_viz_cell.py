import json
import sys

notebook_path = "FIND_Frangi_Fusion_Avignon_Colab.ipynb"

# The new source code for the visualization cell
new_viz_code = [
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from IPython.display import display, Markdown\n",
    "import matplotlib.ticker as ticker\n",
    "\n",
    "# Configuration du style graphique\n",
    "sns.set_theme(style=\"whitegrid\", context=\"notebook\", font_scale=1.1)\n",
    "\n",
    "def display_noise_results(df):\n",
    "    if df is None or df.empty:\n",
    "        print(\"Le DataFrame est vide ou non défini.\")\n",
    "        return\n",
    "\n",
    "    # 1. TABLEAU SYNTHÉTIQUE\n",
    "    display(Markdown(\"### 📊 Synthèse Numérique (Moyenne ± Écart-type)\"))\n",
    "    metrics_map = {\n",
    "        \"Jaccard\": \"Jaccard\",\n",
    "        \"Tversky\": \"Tversky\",\n",
    "        \"Wasserstein\": \"Wasserstein\",\n",
    "        \"CSD_Jaccard\": \"CSD Jac.\",\n",
    "        \"CSD_Tversky\": \"CSD Tvs.\",\n",
    "        \"CSD_Wasserstein\": \"CSD Wass.\"\n",
    "    }\n",
    "    avail_cols = [c for c in metrics_map.keys() if c in df.columns]\n",
    "    grouped = df.groupby(['NoiseExp', 'NoiseLevel'])[avail_cols].agg(['mean', 'std'])\n",
    "    summary_df = pd.DataFrame(index=grouped.index)\n",
    "    for col in avail_cols:\n",
    "        short_name = metrics_map[col]\n",
    "        mean_col = grouped[col]['mean']\n",
    "        std_col = grouped[col]['std']\n",
    "        summary_df[short_name] = mean_col.apply(lambda x: f\"{x:.3f}\" if pd.notnull(x) else \"-\") + \\\n",
    "                                 \" ± \" + \\\n",
    "                                 std_col.apply(lambda x: f\"{x:.3f}\" if pd.notnull(x) else \"-\")\n",
    "    display(summary_df.style.set_properties(**{'text-align': 'center'}).set_table_styles([\n",
    "        dict(selector='th', props=[('text-align', 'center')])\n",
    "    ]))\n",
    "\n",
    "    # 2. VISUALISATION GRAPHIQUE\n",
    "    display(Markdown(\"### 📈 Courbes de Robustesse (Figures Individuelles)\"))\n",
    "\n",
    "    experiments = df['NoiseExp'].unique()\n",
    "    \n",
    "    # Configuration des métriques\n",
    "    # ColName, CSDColName, YLim, YStep\n",
    "    metrics_config = [\n",
    "        (\"Jaccard\", \"CSD_Jaccard\", (0, 1), 0.1),\n",
    "        (\"Tversky\", \"CSD_Tversky\", (0, 1), 0.1),\n",
    "        (\"Wasserstein\", \"CSD_Wasserstein\", (0, 50), 10)\n",
    "    ]\n",
    "\n",
    "    for exp in experiments:\n",
    "        subset = df[df['NoiseExp'] == exp]\n",
    "        if subset.empty: continue\n",
    "        \n",
    "        display(Markdown(f\"#### Expérience : {exp}\"))\n",
    "\n",
    "        for (our_col, csd_col, ylims, ystep) in metrics_config:\n",
    "            \n",
    "            # Création de la figure\n",
    "            fig, ax = plt.subplots(figsize=(6, 5))\n",
    "            \n",
    "            # Tracé Ours\n",
    "            sns.lineplot(\n",
    "                data=subset, x=\"NoiseLevel\", y=our_col,\n",
    "                label=\"Ours (Frangi)\",\n",
    "                color=\"#d62728\", marker=\"o\", linewidth=2.5, errorbar='sd', ax=ax\n",
    "            )\n",
    "\n",
    "            # Tracé CSD\n",
    "            if csd_col in subset.columns and subset[csd_col].notna().any():\n",
    "                sns.lineplot(\n",
    "                    data=subset, x=\"NoiseLevel\", y=csd_col,\n",
    "                    label=\"CrackSegDiff\",\n",
    "                    color=\"#2ca02c\", marker=\"s\", linestyle=\"--\", linewidth=2, errorbar='sd', ax=ax\n",
    "                )\n",
    "\n",
    "            # --- Configuration des Axes ---  \n",
    "            # Pas de titre\n",
    "            ax.set_title(\"\")\n",
    "            \n",
    "            # Axe X : 0 à 0.5, pas 0.1\n",
    "            ax.set_xlim(0, 0.5)\n",
    "            ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))\n",
    "            ax.set_xlabel(\"\") # Pas de label d'axe\n",
    "            ax.set_xticklabels([]) # Pas de graduation (chiffres)\n",
    "            \n",
    "            # Axe Y\n",
    "            ax.set_ylim(*ylims)\n",
    "            ax.yaxis.set_major_locator(ticker.MultipleLocator(ystep))\n",
    "            ax.set_ylabel(\"\") # Pas de label d'axe\n",
    "            ax.set_yticklabels([]) # Pas de graduation (chiffres)\n",
    "            \n",
    "            # Grille en pointillés\n",
    "            ax.grid(True, linestyle='--', alpha=0.7)\n",
    "            \n",
    "            # Légende\n",
    "            ax.legend(loc=\"best\", frameon=True)\n",
    "\n",
    "            plt.tight_layout()\n",
    "            plt.show()\n",
    "\n",
    "if 'df_noise' in locals():\n",
    "    display_noise_results(df_noise)\n",
    "else:\n",
    "    print(\"La variable 'df_noise' n'est pas définie.\")\n"
]

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Locate the cell with 'display_noise_results'
    found = False
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source_code = "".join(cell['source'])
            if "def display_noise_results" in source_code:
                nb['cells'][i]['source'] = new_viz_code
                found = True
                break
    
    if found:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=2, ensure_ascii=False)
        print("Notebook visualization cell updated.")
    else:
        print("Could not find the visualization cell.")

except Exception as e:
    print(f"Error: {e}")
