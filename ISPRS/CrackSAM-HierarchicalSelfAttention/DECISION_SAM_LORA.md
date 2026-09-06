# Perspective : SAM gelé + LoRA + proximité ultramétrique Frangi

**Choix du 6 septembre 2026 : ajouter un biais hiérarchique doux dans une seule attention globale de SAM 2, pendant l’apprentissage des mêmes LoRA que la baseline.** C’est mon meilleur compromis pour une proposition en une slide. Aucun résultat ne permet encore de le déclarer meilleur que SAM + LoRA.

## L’hypothèse précise

Frangi peut surdétecter les ombres et la granularité tout en fournissant des relations utiles. **Détecter une fissure et rapprocher deux fragments d’une même fissure sont deux problèmes différents.** Le rôle sémantique reste assuré par SAM ; le graphe propose une organisation des échanges.

Sur un graphe fixé de coûts d, pour deux candidats distincts d’une même composante, la première hauteur de fusion vérifie :

$$
u_{ij}=\min_{P:i\leadsto j}\max_{e\in P}d_e
=\max_{e\in\operatorname{chemin}_{\mathrm{MST}}(i,j)}d_e.
$$

Poser `u_ii = 0`. Les composantes déconnectées sont traitées séparément.

Une longue chaîne de raccords compatibles peut donc rapprocher des fragments éloignés spatialement. Une ombre voisine peut rester distante dans l’arbre. C’est le mécanisme maximin utilisé notamment par [MALIS, Turaga et al., NIPS 2009](https://proceedings.neurips.cc/paper_files/paper/2009/file/68d30a9594728bc39aa24be94b319d21-Paper.pdf) pour relier affinités et connexité après seuillage.

**« Même fissure ⇒ petit u » reste une hypothèse sur les données.** Il faut une couverture suffisante et un chemin sans mauvais raccord. Une interruption peut faire monter u ; une chaîne parasite peut au contraire rapprocher fissure et fond. Des relations ombre–ombre ne sont pas nécessairement mauvaises : elles peuvent transmettre un contexte de fond. Le danger est surtout le mélange fissure–fond, l’attraction par de grands groupes et les collisions entre branches dans les tokens.

## Ce que les essais imposent

Dans [GeoLoRA, 1695 images](../CrackSAM-GeoLoRA/tables/generated/), l’IoU stricte moyenne par image vaut 0,62763 avec la perte tolérante seule, 0,62698 avec les cartes géométriques et 0,62655 avec les cartes permutées à l’entraînement. **Aucun bénéfice convaincant des descripteurs.** Les [onze cartes corrigent les sorties de l’encodeur](../CrackSAM-GeoLoRA/geolora/adapter.py), avec LoRA coentraîné ; elles ne transmettent pas de groupes emboîtés. La [réserve sur leur alignement à l’entraînement](RECHERCHES.md#une-réserve-sur-linterprétation-causale) limite l’explication causale, sans justifier une nouvelle carte locale.

La [perte clDice déjà testée](../CrackSAM-GeoLoRA/tables/generated/eval_cldice.json) augmente la couverture du squelette mais abaisse l’IoU à 0,60659, contre 0,62414 pour sa baseline. L’[arbitre GFA](../CrackSAM-GFA/RAPPORT.md) atteint 0,61714 contre 0,62375 pour sa référence : il utilise un ensemble plat de fragments, sans dendrogramme. Les [prompts denses et correcteurs raster](../CrackSAM/docs/08_AUDIT_CRACKSAM2_FRANGIGRAPH_LORA.md) ne démontrent pas davantage un gain propre à Frangi. Les protocoles et agrégations doivent rester distincts.

## Comparaison des pistes

| Proposition | Décision pour la demande actuelle |
|---|---|
| Cartes Hessiennes, orientation, contraste ; nouveaux prompts | Écarter : information locale déjà testée sans gain convaincant. |
| Loss de lissage dans les groupes | Écarter : transforme les faux raccords en contraintes sur la prédiction. |
| Loss contrastive avec paires validées par le GT | Plus saine, mais Frangi devient une stratégie d’échantillonnage ; avantage propre non établi. |
| SAM vérifie/nettoie Frangi puis arbitre le masque | Les variantes raster ou fragments plats n’ont pas convaincu ; ajoute une tête ou une passe. |
| Lecteur extérieur, Tree-LSTM, SPT, ondelettes | Programme polyèdres 3D intéressant, mais déplace l’apprentissage vers un autre lecteur. |
| Fusion de tokens ou partage contraint des attentions HSA | Peut supprimer des différences utiles entre extrémités et jonctions. |
| Biais ajouté après avoir gelé aussi LoRA | Simple, mais l’adaptation n’a pas appris en présence du signal. |
| Moyennes hiérarchiques dans les canaux LoRA | Économique, mais mélange les voisins avant de vérifier leur compatibilité visuelle. |
| **Biais doux pendant l’apprentissage de LoRA** | **Retenu : relations hiérarchiques et scores visuels se combinent pour chaque paire.** |

[Conv-LoRA, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/71914867b30fd52452dd0129d1ddbed5-Paper-Conference.pdf), fournit un précédent pour des opérations spatiales dans l’espace de faible rang de SAM. Remplacer ses convolutions par des moyennes Frangi reste une hypothèse alternative si le biais dense est trop coûteux.

## Le mécanisme retenu

Construire les regroupements avec les coûts EUVIP `d_ij = min(1, ρ_ij(1−S⁰_ij))`, sur un support fixé **avant sélection de la plus grande composante et élagage**. L’extracteur ISPRS historique utilise des coûts non bornés : reconstruire explicitement la convention du papier.

Poser `κ_ij = 1−u_ij`, zéro entre composantes déconnectées. κ mesure la fraction des seuils de `[0,1]` où les candidats sont réunis. Projeter vers les tokens par `P κ Pᵀ`, avec P non négatif et chaque ligne couverte de somme un. Neutraliser diagonale et interactions non couvertes pour obtenir `B_H`. **u satisfait l’inégalité ultramétrique dans chaque composante, avec des fusions à zéro possibles ; la projection moyenne ne préserve pas généralement cette propriété.**

Dans un seul bloc global :

$$
\operatorname{Attention}_H=
\operatorname{softmax}\!\left(
\frac{Q_{\mathrm{LoRA}}K^\top}{\sqrt{d_h}}+\beta B_H
\right)V_{\mathrm{LoRA}}.
$$

Les poids préentraînés restent gelés. Les [LoRA locales adaptent les projections Q et V](../CrackSAM/cracksam2/model.py), pas la projection K ; les clés peuvent évoluer indirectement avec les couches précédentes. Ajouter un seul β partagé entre têtes, initialisé à zéro et appris avec projection sur `[0,1]`. Cette borne est une convention de premier essai : le bonus ne dépasse pas un logit. Un bonus d’attention **n’est pas un bonus de probabilité de fissure**. Les scores visuels peuvent le contrebalancer, sans garantie automatique de rejet des mauvais liens.

**Référence principale : [Graphormer, Ying et al., NeurIPS 2021, §3.1.2, équation 6](https://arxiv.org/html/2106.05234v5#S3.SS1.SSS2).** Il ajoute aux logits un biais fondé sur une relation de graphe. La hauteur de fusion Frangi et l’apprentissage limité à LoRA + β sont notre transposition.

## Faisabilité et décision expérimentale

Le dernier bloc global Hiera-L, indice 43, convient à un biais image–image ; le décodeur standard n’a pas cette self-attention. À 4096 tokens, B en FP16 occupe 32 Mio par image ; huit matrices de scores FP32 matérialisées occuperaient 512 Mio. Vérifier le backend. Les LoRA de l’encodeur apprennent : ses caractéristiques ne peuvent pas rester en cache pendant cet entraînement.

**Avant l’entraînement, vérifier la complémentarité relationnelle.** À distance spatiale comparable, les petits u relient-ils des pixels de fissure correctement reconnus aux erreurs résiduelles de SAM, sans multiplier les liens vers le fond ? Comparer proximité spatiale et similarités SAM, notamment dans les ombres et zones granuleuses. Le GT binaire distingue fissure/fond ; il ne fournit pas forcément les instances nécessaires pour affirmer « même fissure ». Les caches locaux ne contiennent pas les arêtes pondérées permettant ce diagnostic : un export du graphe est requis.

Puis comparer absence de biais, support Frangi plat, adjacence locale et hiérarchie, avec mêmes initialisation LoRA, annotations, pertes, augmentations alignées et budget. Le graphe doit correspondre à l’image effectivement présentée ; ne pas conserver des chemins sortant d’un recadrage. Une hiérarchie spatiale ou visuelle sur les mêmes candidats teste l’intérêt propre de Frangi. Si l’on repart du checkpoint actuel, continuer aussi la baseline pendant le même nombre de mises à jour. Mesurer IoU/Dice, ruptures et faux raccords, sur scènes séparées et plusieurs graines. β=0 neutralise le biais aux poids courants ; il ne restitue pas le checkpoint initial après coentraînement.

**Mon niveau de confiance : hypothèse relationnelle plausible, gain non établi.** Les échecs de détection Frangi ne réfutent pas cette hypothèse ; ils ne disparaissent pas non plus par passage à l’ultramétrique. Je ne prédis pas un gain global. Si le diagnostic et les comparaisons n’apportent aucun avantage, conserver SAM + LoRA et conclure que Frangi-graphe n’aide pas dans ce cadre.
