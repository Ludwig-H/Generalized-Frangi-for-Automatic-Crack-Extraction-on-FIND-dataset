# Une slide : SAM gelé + LoRA + hiérarchie Frangi-graphe

**Titre : « Perspective — guider les échanges de SAM par la proximité ultramétrique »**

![Principe du guidage](figures/guidage_hierarchique.png)

À afficher :

- **SAM gelé + LoRA** reconnaît les fissures et le fond.
- **Frangi-graphe** rapproche des fragments reliés par une chaîne compatible.
- **Un bonus d’attention** favorise leurs échanges, selon un poids appris β.
- **Apprentissage : LoRA + β**, avec les mêmes annotations que la baseline.

$$
\text{attention}=\operatorname{softmax}
\bigl(\text{scores visuels}+\beta\,\text{proximité hiérarchique}\bigr).
$$

À dire, environ 35 secondes :

> Frangi surdétecte les ombres et les textures, alors que SAM les distingue mieux. L’idée est d’exploiter une autre propriété : deux morceaux d’une même fissure peuvent être proches dans la hiérarchie, même s’ils sont éloignés dans l’image. Cette proximité donne un bonus à leurs échanges dans une attention de SAM. Les scores visuels restent présents et LoRA apprend avec ce guidage. Il faudra vérifier que les bonnes relations apportées par le graphe compensent ses faux raccords.

**Référence :** [Ying et al., Graphormer, NeurIPS 2021](https://arxiv.org/abs/2106.05234), pour le biais relationnel dans l’attention. La proximité ultramétrique Frangi et l’adaptation LoRA sont notre transposition.

Les [définitions et limites](DECISION_SAM_LORA.md) restent hors slide. Ce premier essai K=1 ne valide pas encore le programme des polyèdres LiDAR K≥2.
