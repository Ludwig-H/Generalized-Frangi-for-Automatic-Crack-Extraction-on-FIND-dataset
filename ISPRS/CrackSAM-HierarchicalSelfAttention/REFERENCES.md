# Références et idées reprises

**Graphormer justifie l’insertion ; MALIS éclaire la hiérarchie.** Leur combinaison avec Frangi et SAM reste notre proposition.

| Référence | Ce que nous reprenons et sa limite |
|---|---|
| **Hauseux et al., EUVIP 2026**, [*Multi-Modal, Training-Free Crack Extraction via Generalized Frangi Graph*](../../EUVIP/EUVIP_2026_Generalized_Frangi_Multimodality_camera-ready.pdf). | Candidats, alignement et coûts du graphe ; l’arbre sert à extraire un squelette, sans guider SAM. |
| **Ying et al., NeurIPS 2021**, [*Do Transformers Really Perform Badly for Graph Representation?*](https://papers.nips.cc/paper_files/paper/2021/hash/f1c1592588411002af340cbaedd6fc33-Abstract.html) — Graphormer. | Un biais relationnel ajouté avant softmax (§3.1.2, équation 6) ; leur relation repose sur les plus courts chemins, sans hiérarchie Frangi. |
| **Turaga et al., NIPS 2009**, [*Maximin affinity learning of image segmentation*](https://papers.nips.cc/paper_files/paper/2009/hash/68d30a9594728bc39aa24be94b319d21-Abstract.html) — MALIS. | Le lien entre affinité maximin, connexité après seuillage et arbre couvrant (§4) ; nous ne reprenons ni sa perte ni l’apprentissage des affinités. |
| **Hu et al., ICLR 2022**, [*LoRA: Low-Rank Adaptation of Large Language Models*](https://www.microsoft.com/en-us/research/publication/lora-low-rank-adaptation-of-large-language-models/). | Des corrections de faible rang apprises avec les poids préentraînés gelés ; la hiérarchie n’appartient pas à LoRA. |
| **Ravi et al., ICLR 2025** (prépublication 2024), [*SAM 2: Segment Anything in Images and Videos*](https://proceedings.iclr.cc/paper_files/paper/2025/hash/45c1f6a8cbf2da59ebf2c802b4f742cd-Abstract-Conference.html). | Le modèle de segmentation préentraîné ; son architecture ne fournit pas le guidage métier proposé ici. |

## Notre transposition

Le graphe fixe le biais de chaque image. **LoRA et un seul coefficient β apprennent ensemble** ; les poids préentraînés restent gelés. L’ultramétrique concerne les candidats ; sa moyenne vers les tokens ne conserve généralement pas cette propriété.

## Citations sur les figures

- **Figures TikZ 1–2** : « Graphe : Hauseux et al., EUVIP 2026 ; lien connexité–arbre : Turaga et al., 2009. »
- **Figure TikZ 3** : « Biais : Graphormer, 2021 ; adaptation : LoRA, 2022 ; modèle : SAM 2, 2025. »

Entrées BibTeX : [references.bib](references.bib).
