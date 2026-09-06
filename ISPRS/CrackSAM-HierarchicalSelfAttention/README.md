# Perspective : SAM gelé + LoRA + hiérarchie Frangi-graphe

**SAM reconnaît les fissures ; Frangi-graphe propose quels fragments mettre en relation.** On ajoute un biais hiérarchique dans une attention de SAM 2, pendant l’apprentissage de LoRA. Le gain recherché concerne les fissures fragmentées ; il reste à vérifier.

![SAM gelé, LoRA et guidage hiérarchique](figures/03_sam_lora.png)

## Pourquoi une hiérarchie ?

Frangi surdétecte les ombres et la granularité. Pourtant, deux morceaux éloignés d’une fissure peuvent fusionner tôt dans son arbre : une chaîne de raccords compatibles les rapproche. **La hauteur de fusion décrit cette relation globale.** Elle organise l’information géométrique sans fournir une nouvelle observation de l’image.

Des relations ombre–ombre peuvent apporter un contexte de fond. Le risque est surtout une fusion précoce fissure–fond ; les scores visuels de SAM doivent rester présents.

## Du graphe au biais d’attention

**1. Construire les regroupements.** Reprendre les candidats et les coûts [EUVIP](../../EUVIP/LaTeX/main.tex), avant sélection de composante et élagage. Intensité et forme prolongent Frangi ; l’alignement caractérise les relations du graphe.

$$d_{ij}=\min\{1,\rho_{ij}(1-S_{ij}^{(0)})\}.$$

Ici, $\rho_{ij}$ est la distance entre candidats et $S_{ij}^{(0)}$ leur compatibilité multiscalaire du papier. Pour deux candidats distincts d’une même composante, la hauteur de première fusion est le coût maximal sur leur chemin $T_{ij}$ dans l’arbre couvrant minimal :

$$u_{ij}=\max_{e\in T_{ij}}d_e,\qquad \kappa_{ij}=1-u_{ij}.$$

Poser $u_{ii}=0$ et $\kappa_{ij}=0$ entre composantes déconnectées. Fusion précoce signifie proximité forte, sans choisir une coupe unique.

**2. Passer aux tokens.** Chaque token moyenne les candidats qu’il couvre. Avec ces poids dans $P$, calculer $P\kappa P^\top$, puis annuler diagonale et interactions non couvertes : on obtient $B_H$. Les lignes couvertes de $P$ somment à un. Cette moyenne ne conserve généralement pas l’ultramétrie des candidats.

**3. Favoriser les échanges.** Dans une seule attention globale de l’encodeur :

$$A_H=\mathrm{softmax}\left(Q_{\mathrm{LoRA}}K^\top/\sqrt{d_h}+\beta B_H\right),\qquad Y=A_HV_{\mathrm{LoRA}}.$$

Les poids préentraînés restent gelés. Les [LoRA existantes](../CrackSAM/cracksam2/model.py) adaptent les projections Q et V. Un seul $\beta$, partagé entre têtes, apprend avec elles : départ à zéro, projection dans $[0,1]$ après chaque mise à jour. Le biais favorise un échange, sans imposer une étiquette de fissure. [Graphormer](REFERENCES.md) fournit ce principe d’insertion dans l’attention.

> **Note : confiance locale (piste ultérieure).** Sans modifier le premier test, on pourrait remplacer $\beta(B_H)_{ij}$ par $\beta\,g_i(I)g_j(I)(B_H)_{ij}$, avec $g_i(I)\in[0,1]$ prédit par un petit module à partir des caractéristiques visuelles avant l’attention guidée et d’un résumé global de l’image. Ce module apprendrait avec LoRA via la perte de segmentation ; la hiérarchie resterait calculée. L’objectif serait d’atténuer localement un guidage peu fiable, notamment près de certaines ombres, sans désactiver le guidage sur toute l’image. Ces coefficients ne seraient ni des probabilités de fissure ni une évaluation explicite des chemins. Variante non implémentée, à comparer au $\beta$ global ; son bénéfice reste à vérifier.

## Le premier test

Comparer **SAM + LoRA**, **biais de proximité spatiale** et **biais hiérarchique**, sur les mêmes candidats et avec le même budget. Garder annotations, pertes et augmentations alignées ; reconstruire les chemins après recadrage. Mesurer IoU, ruptures et faux raccords, notamment dans les ombres. Les [anciens essais négatifs](ARCHIVES.md) motivent ce contrôle.

Point de départ technique : bloc global 43 de Hiera-L. À 4096 tokens, le biais FP16 coûte 32 Mio par image, hors calcul d’attention : vérifier la mémoire. Ce dossier présente la méthode ; il ne contient pas son implémentation SAM.

## Pour la soutenance

- [Une slide et son texte oral](SOUTENANCE.md).
- [Trois figures TikZ réutilisables et leur PDF](figures/README.md).
- [Références : idées reprises et limites](REFERENCES.md), avec [BibTeX](references.bib).
- [Archive unique des anciennes pistes](ARCHIVES.md).
