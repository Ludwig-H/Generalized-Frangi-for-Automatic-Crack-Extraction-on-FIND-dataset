# Roadmap : Migration de CrackSAM vers SAM 2 avec guidage par Frangi-Graphe (GPU)

> [!IMPORTANT]
> **Consignes initiales pour ChatGPT** :
> 1. Avant de commencer à coder quoi que ce soit, lis attentivement le papier scientifique de référence fourni : [CrackSAM.pdf](file:///workspaces/Generalized-Frangi-for-Automatic-Crack-Extraction-on-FIND-dataset/ISPRS/CrackSAM/CrackSAM.pdf). Ce papier détaille la méthodologie de fine-tuning PEFT (LoRA/Adapter) et sert de base scientifique à ce projet.
> 2. Dans ce projet, nous ciblons le modèle de fondation **SAM 2** disponible sur le dépôt officiel : [Meta Research - SAM 2](https://github.com/facebookresearch/sam2). 
> 3. Note que le passage de SAM 1 à SAM 2 nécessitera **plusieurs adaptations importantes au niveau du code** :
>    * La bibliothèque Python à utiliser est `sam2` (et non plus `segment_anything`).
>    * Les classes et fonctions d'initialisation du modèle (`build_sam2`, `SAM2ImagePredictor`) diffèrent légèrement de SAM 1.
>    * Le backbone d'image de SAM 2 est **Hiera** (au lieu du ViT plat de SAM 1). Les couches de projection de l'attention à cibler pour LoRA ne porteront pas les mêmes noms (ex. cibles dans les blocs d'attention de Hiera).

Ce document constitue un plan de route détaillé destiné à guider le codage de l'implémentation de **SAM 2** adapté aux fissures et guidé géométriquement par la similarité de **Frangi-Graphe**.

---

## 1. Contexte du projet et Fichiers Clés

Le dépôt actuel contient les dossiers et fichiers suivants qu'il faudra exploiter :
*   **Code originel CrackSAM (SAM 1 + Adapters/LoRA)** : [ISPRS/CrackSAM/CrackSAM/CrackSAM/](file:///workspaces/Generalized-Frangi-for-Automatic-Crack-Extraction-on-FIND-dataset/ISPRS/CrackSAM/CrackSAM/CrackSAM)
    *   [train.py](file:///workspaces/Generalized-Frangi-for-Automatic-Crack-Extraction-on-FIND-dataset/ISPRS/CrackSAM/CrackSAM/CrackSAM/train.py) : Script d'entraînement principal.
    *   [test.py](file:///workspaces/Generalized-Frangi-for-Automatic-Crack-Extraction-on-FIND-dataset/ISPRS/CrackSAM/CrackSAM/CrackSAM/test.py) : Script d'évaluation.
    *   [trainer.py](file:///workspaces/Generalized-Frangi-for-Automatic-Crack-Extraction-on-FIND-dataset/ISPRS/CrackSAM/CrackSAM/CrackSAM/trainer.py) : Boucle d'entraînement.
    *   [datasets/dataset_khanhha.py](file:///workspaces/Generalized-Frangi-for-Automatic-Crack-Extraction-on-FIND-dataset/ISPRS/CrackSAM/CrackSAM/CrackSAM/datasets/dataset_khanhha.py) : Dataloader pour le jeu de données d'entraînement.
    *   [utils.py](file:///workspaces/Generalized-Frangi-for-Automatic-Crack-Extraction-on-FIND-dataset/ISPRS/CrackSAM/CrackSAM/CrackSAM/utils.py) : Fonctions de perte (Dice, Focal) et calcul des métriques de segmentation.
*   **Algorithme Frangi-Graphe optimisé GPU** :
    *   [test_k2_clean.py](file:///workspaces/Generalized-Frangi-for-Automatic-Crack-Extraction-on-FIND-dataset/test_k2_clean.py#L73) : Contient la classe `FrangiHessianGPU` (lignes 73-137) et la fonction `extract_frangi_graph_gpu` (lignes 140-710).
*   **Environnement GCP (RTX PRO 6000 Blackwell)** :
    *   [gcp-migration/README.md](file:///workspaces/Generalized-Frangi-for-Automatic-Crack-Extraction-on-FIND-dataset/gcp-migration/README.md) : Instructions pour déployer et configurer la machine GPU virtuelle.

---

## 2. Configuration des Jeux de Données (Datasets)

Pour reproduire exactement l'évaluation du papier **CrackSAM.pdf**, le protocole doit intégrer les ensembles de données suivants :

### A. Dataset principal d'entraînement et de validation (Khanhha Dataset)
*   **Nom** : Khanhha Crack Segmentation Dataset (9 603 images d'entraînement, 1 695 images de test).
*   **Lien** : [https://github.com/khanhha/crack_segmentation](https://github.com/khanhha/crack_segmentation)
*   **Dossiers** :
    *   `trainingset/` (images et masques) référencé par `lists/lists_khanhha/train.txt`.
    *   `validationset/` (images et masques) référencé par `lists/lists_khanhha/val_vol.txt`.

### B. Ensembles d'évaluation (Test & Robustesse)
L'évaluation finale du modèle doit se faire sur les 6 configurations de test définies dans le papier :
1.  **Test Set (Original)** : Le split de test propre de Khanhha (1 695 images) référencé par `lists/lists_khanhha/test_vol.txt`.
2.  **Noisy Test Set 1 (Luminosité réduite + Flou)** :
    *   *Perturbation* : Convertir les images de test de Khanhha dans l'espace colorimétrique HSV, soustraire 50 au canal V (Value) pour simuler une faible luminosité, repasser en RGB, puis appliquer un flou gaussien avec un noyau de $9 \times 9$.
3.  **Noisy Test Set 2 (Flou sévère + Sous-échantillonnage)** :
    *   *Perturbation* : Appliquer un flou gaussien avec un noyau de $21 \times 21$, sous-échantillonner à la moitié de la taille originale (par interpolation cubique), puis ré-échantillonner (redimensionner) à $448 \times 448$.
4.  **Road420 (Zero-Shot)** : 420 images réelles de fissures sur route avec interférences (ombres, marquages).
    *   *Lien de téléchargement* : [Google Drive - 1khUfS2uDZb5eDOhpL1qJPYsOxso7Limu](https://drive.google.com/file/d/1khUfS2uDZb5eDOhpL1qJPYsOxso7Limu/view)
5.  **Facade390 (Zero-Shot)** : 390 images de fissures sur murs extérieurs capturées par drone.
    *   *Lien de téléchargement* : [Google Drive - 1P1b15kRQpVcT7cNDzZB_1vFTrN0WKPB_](https://drive.google.com/file/d/1P1b15kRQpVcT7cNDzZB_1vFTrN0WKPB_/view)
6.  **Concrete3k (Zero-Shot)** : 3 000 paires d'images de fissures de béton.
    *   *Lien de téléchargement* : [https://github.com/CHDyshli/HrSegNet4CrackSegmentation](https://github.com/CHDyshli/HrSegNet4CrackSegmentation)

---

## 3. Choix d'Architecture LoRA et Hyperparamètres d'Entraînement

Pour respecter scrupuleusement la configuration validée dans **CrackSAM.pdf**, implémenter les hyperparamètres et choix d'architecture suivants :

### A. Cibles de LoRA sur SAM 2
*   **Emplacements** : Appliquer LoRA uniquement sur les projections de **requête (query - q)** et de **valeur (value - v)** au sein des couches d'attention du visual encoder (Hiera) et du mask decoder.
    *   *Note du papier* : L'ablation (Table 3) montre qu'appliquer LoRA sur toutes les projections (`qkvo`) améliore légèrement le test set propre mais **dégrade la généralisation zero-shot** (sur Road420/Facade390). La configuration `qv` est donc requise pour garantir la robustesse trans-domaine.
*   **Rang (Rank)** : Configurer $r = 4$ ou $r = 8$ (Table 2 & Table 4 du papier).

### B. Hyperparamètres d'optimisation
*   **Loss Function** : Somme pondérée d'une Cross-Entropy (CE) et d'une Dice Loss :
    $$L = \lambda L_{CE} + (1 - \lambda) L_{Dice} \quad \text{avec} \quad \lambda = 0.2$$
*   **Optimiseur** : `AdamW` avec $\beta_1 = 0.9$, $\beta_2 = 0.999$, et `weight_decay = 0.01`.
*   **Taux d'apprentissage (Learning Rate)** :
    *   LR initial maximal : `0.0004`.
    *   **Scheduler** : Planificateur de type `poly` précédé d'un échauffement linéaire (*linear warmup*) :
        *   *Warmup* : Augmentation linéaire de 0 à `0.0004` durant les **300 premières itérations**.
        *   *Poly Decay* : Décroissance selon la formule $(1 - \frac{\text{iter} - \text{warmup}}{\text{max\_iter} - \text{warmup}})^{\text{power}}$ avec **$\text{power} = 6$**.
*   **Nombre d'Époques (Epochs)** : 140 époques.
*   **Batch Size** : 8.
*   **Seuil de binarisation du masque** : 0.5.
*   **Augmentations de données** : Rotations aléatoires et retournements horizontaux/verticaux (déjà présents dans le dataloader).

---

## 4. Plan de Comparaison Expérimentale

Nous voulons comparer deux configurations distinctes :

### Configuration 1 : Baseline SAM 2 + LoRA (Sans Frangi)
1.  **Architecture** : Utiliser **SAM 2** (modèle `sam2_hiera_large.pt` de Meta).
2.  **Fine-Tuning LoRA** : Appliquer LoRA sur `q` et `v` (encodeur Hiera + décodeur de masques).
3.  **Entraînement/Validation** : Selon les paramètres définis dans la Section 3.
4.  **Inférence** : Prédiction de masques classique sans prompt externe sur les 6 ensembles de test.

### Configuration 2 : SAM 2 + LoRA + Guidage Frangi-Graphe (Notre Méthode)
1.  **Génération de la Carte Frangi-Graphe sur GPU** :
    *   Exécuter la fonction `extract_frangi_graph_gpu` avec un canal unique en entrée (ex. `imgs = {'visible': image}`) et un poids de 1.0 (`weights = {'visible': 1.0}`).
    *   **Paramètres par défaut obligatoires** :
        *   `K = 1` (graphe de cliques d'arêtes simples, MST classique).
        *   `Σ = [1.0, 3.0, 5.0, 9.0, 15.0]` (multi-échelle).
        *   `R = 3` (rayon de voisinage).
    *   La fonction retourne la carte `similarity_img` (valeurs réelles dans $[0, 1]$ issues de `node_sim_max`).
2.  **Transformation en Pseudo-Logits** :
    *   Transformer la carte de probabilités $P$ (similarité) en logits $L$ :
        $$L = \log\left(\frac{P}{1-P}\right)$$
    *   *Note d'implémentation* : Écrêter $P$ sur l'intervalle $[\epsilon, 1-\epsilon]$ avec $\epsilon = 10^{-5}$ avant le calcul pour éviter le logarithme de 0 ou la division par 0.
3.  **Prompt Spatial de SAM 2** :
    *   Redimensionner la carte de logits à la dimension $256 \times 256$.
    *   Injecter ce tenseur de dimension `(B, 1, 256, 256)` comme argument `mask_input` lors des passes avant (*forward*) d'entraînement et de test de SAM 2. Le calcul du graphe de Frangi n'est pas différentiable, il agit donc comme un tenseur de prompt statique (pas de gradients rétropropagés à travers Frangi-Graphe).
4.  **Fine-Tuning LoRA** : Les gradients sont rétropropagés uniquement à travers les poids LoRA de l'encodeur d'images et du décodeur de masques de SAM 2.

---

## 5. Métriques d'Évaluation

ChatGPT doit coder l'évaluation sur l'ensemble de test en calculant les métriques suivantes (définies dans [utils.py](file:///workspaces/Generalized-Frangi-for-Automatic-Crack-Extraction-on-FIND-dataset/ISPRS/CrackSAM/CrackSAM/CrackSAM/utils.py#L96) et [test_k2_clean.py](file:///workspaces/Generalized-Frangi-for-Automatic-Crack-Extraction-on-FIND-dataset/test_k2_clean.py#L734)) :
1.  **Précision (Precision)**
2.  **Rappel (Recall)**
3.  **Dice Coefficient (F1-Score)**
4.  **IoU / Jaccard**
5.  **Wasserstein Distance sur Masque Direct** :
    *   Appliquer la distance de Wasserstein (EMD) **directement sur les masques obtenus** (sans squelettisation préalable).
    *   Normaliser les masques pour les traiter comme des distributions de probabilité discrètes dans $\mathbb{R}^2$ (la somme des poids de chaque masque doit être égale à 1).
    *   Utiliser la bibliothèque `POT` (Python Optimal Transport) via `ot.emd2` sur les coordonnées spatiales $(y, x)$ des pixels actifs du masque pondérés par leur valeur normalisée.

---

## 6. Exécution et Déploiement sur GPU Blackwell (GCP)

Pour exécuter le code sur la VM Google Cloud Platform dotée d'une carte NVIDIA RTX PRO 6000 Blackwell (96 Go de VRAM), ChatGPT devra générer des scripts ou fournir les commandes suivantes :

1.  **Initialisation du projet GCP** :
    ```bash
    export GCP_PROJECT_ID="devpod-gpu-exploration"
    gcloud config set project "${GCP_PROJECT_ID}"
    ```
2.  **Déploiement et Démarrage de la VM** :
    ```bash
    # Déploiement de l'instance si inexistante
    ./gcp-migration/deploy.sh
    
    # Démarrage de la VM
    gcloud compute instances start frangi-blackwell-spot --project="${GCP_PROJECT_ID}" --zone="europe-west4-a"
    
    # Connexion SSH
    gcloud compute ssh frangi-blackwell-spot --project="${GCP_PROJECT_ID}" --zone="europe-west4-a"
    ```
3.  **Vérification de l'environnement GPU (Preflight)** :
    Une fois dans la VM, lancer le diagnostic de pré-vol avec coupe-circuit d'arrêt automatique de 240 minutes (4 heures) :
    ```bash
    cd /chemin/vers/dépôt/Generalized-Frangi-for-Automatic-Crack-Extraction-on-FIND-dataset
    ./gcp-migration/blackwell_preflight.sh --arm-shutdown 240
    ```
4.  **Installation des pilotes open-kernel NVIDIA (si requis)** :
    Si le diagnostic signale un problème de pilote open-kernel pour l'architecture Blackwell :
    ```bash
    ./gcp-migration/blackwell_preflight.sh --install-open-driver --skip-docker
    sudo reboot
    ```
5.  **Arrêt propre après exécution** :
    Depuis la machine locale pour libérer les ressources :
    ```bash
    ./gcp-migration/stop_and_verify.sh
    ```

---

## 7. État final de l'expérience du 14 juillet 2026

Les deux entraînements SAM 2 ont atteint leur horizon configuré de 70 époques.
La meilleure baseline est le checkpoint de l'époque 20 ; le meilleur checkpoint
Frangi-similarité est celui de l'époque 25, sélectionné uniquement sur le Dice de
validation. Les cinq jalons Frangi conservés ont été réévalués sous un même
contrat sur les 8 895 conditions image/jeu, avec sauvegarde de chaque prédiction.

| Modèle/jalon | IoU macro, 6 jeux | Δ vs baseline best |
|---|---:|---:|
| Baseline best, époque 20 | 0,567465 | — |
| Frangi, époque 20 | 0,555058 | −0,012407 |
| Frangi, époque 25 best | 0,556288 | −0,011177 |
| Frangi, époque 30 | 0,550911 | −0,016554 |
| Frangi, époque 55 | 0,546557 | −0,020908 |
| Frangi, époque 70 | 0,546430 | −0,021035 |
| CrackSAM Adapter d=32 publié (SAM 1) | 0,571817 | comparaison externe |
| CrackSAM LoRA qv r=4 publié (SAM 1) | 0,577950 | comparaison externe |

Le guidage Frangi n'améliore donc pas la moyenne globale dans cette expérience.
Le jalon 25 apporte un léger gain sur Khanhha bruité 1 (+0,002675 IoU), mais
recule notamment sur Khanhha bruité 2 (−0,029241), Facade390 (−0,016429) et
Concrete3k (−0,009691). Les comparaisons au papier restent externes : CrackSAM
utilise SAM 1 ViT-H, alors que cette expérience utilise SAM 2 Hiera Large.

Le calcul Wasserstein exact dense demandé a été pré-scanné pour les six runs.
Avec un budget de 140 Gio, l'intersection théorique contient 8 888/8 895 cas ;
sept cas distincts dépassent le budget sur au moins un run. Toutefois, le
benchmark réel estime 200–300 heures pour les six runs : un cas admissible de
136,56 Gio n'était toujours pas terminé après 16,5 minutes. La tentative a été
arrêtée proprement après 1 182/8 890 cas baseline, sans échec, et son journal est
reprenable. Aucune moyenne partielle n'est présentée comme un résultat final.
Le scan complet, les seuils de couverture et la politique de publication sont
consignés dans
[`results/2026-07-14_wasserstein_feasibility.json`](results/2026-07-14_wasserstein_feasibility.json).

Le rapport chiffré et illustré complet est disponible dans
[`results/frangi_milestone_report/RAPPORT_FRANGI_MILESTONES.md`](results/frangi_milestone_report/RAPPORT_FRANGI_MILESTONES.md).
Il contient les métriques exhaustives, les statistiques appariées, les courbes,
les comparaisons aux deux modèles publiés, les réussites, les échecs, les cas
clairsemés et douze pseudo-logits Frangi bruts. Les poids conservés, leurs
SHA-256 et leurs chemins de restauration figurent dans
[`results/2026-07-14_checkpoint_manifest.json`](results/2026-07-14_checkpoint_manifest.json).

## 8. Orientation recommandée après analyse des échecs

Le test CPU luminance/chrominance et l'analyse du mode d'injection conduisent à
ne pas poursuivre une variante chrominance seule ni une fusion fixe de deux
prompts denses. La suite recommandée, nommée **SafeFrangi**, conserve la
baseline époque 20 gelée et traduit la similarité Frangi en un résidu
neutralisable autour de l'embedding officiel `no_mask`. L'image est encodée
une fois, les branches baseline et Frangi sont décodées séparément, puis une
porte d'abstention globale et spatiale fusionne tardivement leurs logits afin
de pouvoir sélectionner exactement la sortie baseline.

La justification, l'architecture tensorielle, les pertes, les ablations, les
critères go/no-go, la politique de checkpoints et les points d'intégration sont
détaillés dans
[`results/frangi_safe_recommendation/RAPPORT_RECOMMANDATION_SAFE_FRANGI.md`](results/frangi_safe_recommendation/RAPPORT_RECOMMANDATION_SAFE_FRANGI.md).
