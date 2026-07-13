# Roadmap : Migration de CrackSAM vers SAM 2 avec guidage par Frangi-Graphe (GPU)

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

## 3. Plan de Comparaison Expérimentale

Nous voulons comparer deux configurations distinctes :

### Configuration 1 : Baseline SAM 2 + LoRA (Sans Frangi)
1.  **Architecture** : Utiliser **SAM 2** (modèle `sam2_hiera_large.pt` de Meta).
2.  **Fine-Tuning LoRA** : Appliquer l'adaptation LoRA sur les couches d'attention du visual encoder (Hiera) et du mask decoder de SAM 2.
3.  **Entraînement/Validation** : Identique à la boucle originale de CrackSAM (Focal Loss + Dice Loss, batch size 8 ou 12, optimisation via AdamW).
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

## 4. Métriques d'Évaluation

ChatGPT doit coder l'évaluation sur l'ensemble de test en calculant les métriques suivantes (définies dans [utils.py](file:///workspaces/Generalized-Frangi-for-Automatic-Crack-Extraction-on-FIND-dataset/ISPRS/CrackSAM/CrackSAM/CrackSAM/utils.py#L96) et [test_k2_clean.py](file:///workspaces/Generalized-Frangi-for-Automatic-Crack-Extraction-on-FIND-dataset/test_k2_clean.py#L734)) :
1.  **Précision (Precision)**
2.  **Rappel (Recall)**
3.  **Dice Coefficient (F1-Score)**
4.  **IoU / Jaccard**
5.  **Wasserstein Distance sur Squelette** :
    *   Calculer la distance de Wasserstein (EMD) sur les squelettes extraits et épaissis des prédictions et vérités terrains (en utilisant `skeletonize_lee` et `thicken(..., pixels=3)`).
    *   Utiliser la bibliothèque `POT` (Python Optimal Transport) via `ot.emd2` sur les coordonnées spatiales normalisées des pixels actifs du squelette, comme implémenté dans la fonction `wasserstein_distance_skeletons` de [test_k2_clean.py](file:///workspaces/Generalized-Frangi-for-Automatic-Crack-Extraction-on-FIND-dataset/test_k2_clean.py#L752).

---

## 5. Exécution et Déploiement sur GPU Blackwell (GCP)

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
