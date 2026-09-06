# Perspective : SAM gelé + LoRA + hiérarchie Frangi-graphe

**Proposition principale : un biais fondé sur la proximité ultramétrique, dans une seule attention globale de SAM 2, présent pendant l’apprentissage de LoRA.** Les poids préentraînés restent gelés. Aucun gain sur la baseline n’est actuellement démontré.

## Le principe en une slide

![SAM gelé, LoRA et relations hiérarchiques](figures/guidage_hierarchique.png)

**SAM reconnaît le contenu ; Frangi-graphe propose des relations entre fragments.** Deux morceaux éloignés peuvent rester proches dans la hiérarchie si une chaîne de raccords compatibles les relie. Cette proximité ajoute un bonus aux scores d’attention, qui conservent les compatibilités visuelles.

Les mêmes LoRA que la baseline apprennent avec ce guidage, ainsi qu’un seul coefficient β. La [formulation orale](SOUTENANCE.md) tient sur une slide ; le [bilan complet](DECISION_SAM_LORA.md) compare les propositions et définit le test décisif.

## Pourquoi l’ultramétrique mérite un essai

La surdétection des ombres et de la granularité ne signifie pas nécessairement que toutes les relations Frangi sont mauvaises. Un groupe d’ombre peut fournir un contexte de fond. Le risque important est de rapprocher trop tôt fissure et fond, ou de mélanger leurs branches dans les tokens.

Les essais de cartes locales, clDice et d’arbitrage sur fragments plats n’ont pas démontré de gain propre au graphe. Ils n’ont pas testé cette proximité hiérarchique dans l’attention de SAM.

[Graphormer, NeurIPS 2021](https://arxiv.org/abs/2106.05234), fournit le précédent du biais structurel dans l’attention. Sa transposition à Frangi et SAM + LoRA reste une hypothèse.

## Niveau de confiance

Un contexte utile pour des fissures fragmentées est plausible. Il faut vérifier que la hiérarchie apporte de bonnes relations précisément là où SAM se trompe, au-delà des relations spatiales ou visuelles. Si cet avantage n’apparaît pas, conserver **SAM + LoRA sans Frangi**.

## Recherches complémentaires

- [Résultats négatifs et interfaces déjà testées](RECHERCHES.md).
- [Ancienne comparaison avec LoRA également gelée](PISTES_SANS_REENTRAINEMENT.md).
- [Programme des polyèdres LiDAR 3D](VOIE_POLYEDRES.md) et [lecteur multirésolution](LECTURE_MULTIECHELLE.md), pour une perspective plus large.

Le schéma se régénère avec `python figures/make_figure.py` depuis ce dossier.
