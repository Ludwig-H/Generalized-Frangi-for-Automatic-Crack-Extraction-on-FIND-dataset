# CrackSAM 2 avec guidage Frangi-Graphe

Cette implémentation migre l'expérience CrackSAM de SAM 1 vers le checkpoint
officiel `sam2_hiera_large.pt`. Elle compare deux modèles entraînés avec le
même protocole :

- `baseline` : SAM 2 Hiera Large, LoRA uniquement sur `q` et `v`, sans prompt ;
- `frangi` : même modèle et mêmes hyperparamètres, avec la similarité
  Frangi-Graphe comme prompt dense statique.

Le code SAM 1 d'origine reste inchangé dans `CrackSAM/CrackSAM/`.

## Contrat expérimental

- images sources et masques : 448 x 448 ;
- entrée Hiera : normalisation ImageNet et redimensionnement 1024 x 1024 ;
- prompt : similarité Frangi-Graphe, `K=1`, `R=3`,
  `scales=[1,3,5,9,15]`, puis `log(P/(1-P))` avec `eps=1e-5`, 256 x 256 ;
- LoRA : projections `q/v` des 48 blocs Hiera Large et des 7 attentions du
  mask decoder, rang 4 ou 8 ;
- optimisation : AdamW `(0.9, 0.999)`, weight decay 0.01, batch 8, 70 époques ;
- LR : warmup linéaire 300 itérations jusqu'à `4e-4`, puis poly puissance 6 ;
- perte : `0.2 * BCE + 0.8 * Dice`, avec fond/fissure comme les deux classes ;
- sélection : meilleur Dice de validation, seuil final 0.5.

L'horizon initial de la roadmap était 140 époques. Le run final utilise 70
époques, décision expérimentale prise pour SAM 2 afin de limiter l'overfitting ;
le meilleur checkpoint reste sélectionné sur la validation tous les 5 epochs.

## Installation

SAM 2 requiert PyTorch et torchvision fonctionnels sur le GPU avant
l'installation. Sur l'image GCP Blackwell `common-cu129` du dépôt :

```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv
python3 -m venv ~/.venv-cracksam2
source ~/.venv-cracksam2/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu129
SAM2_BUILD_CUDA=0 python -m pip install --no-build-isolation \
  -r ISPRS/CrackSAM/requirements-sam2.txt
```

L'extension CUDA SAM 2 désactivée ici sert au post-traitement optionnel des
composantes connexes. Le forward, l'entraînement et les métriques de ce projet
ne l'utilisent pas.

## Exécution complète

Après le preflight GPU :

```bash
export CRACKSAM2_DATA_ROOT="$HOME/cracksam2-data"
export CRACKSAM2_ARTIFACT_ROOT="$HOME/cracksam2-artifacts"
export CRACKSAM2_PROMPT_ROOT="$HOME/cracksam2-prompts"
export CRACKSAM2_EPOCHS=70
nohup bash ISPRS/CrackSAM/run_full_cracksam2_experiment.sh \
  > "$HOME/cracksam2-run.log" 2>&1 < /dev/null &
echo $! > "$HOME/cracksam2-run.pid"
```

Le script est idempotent : il valide les datasets déjà extraits, reprend un
cache de prompts interrompu et reprend `latest.pt` après une interruption Spot. Il
télécharge les quatre archives puis n'extrait que les images utiles, sans les
versions haute résolution de Road420/Facade390.
Après un redémarrage de la VM, réarmer le coupe-circuit puis relancer exactement
la même commande ; le contrat du checkpoint refuse toute dérive de protocole.

Les caches Frangi ne sont publiés qu'une fois complets. Leur manifeste lie les
fichiers à la liste ordonnée, au bruitage, au redimensionnement et aux paramètres
`scales/R/K/eps`; l'entraînement refuse un cache absent ou incompatible.
Le pré-calcul s'arrête après `node_sim_max` : le chemin est testé bit à bit
contre la fonction complète et évite seulement le MST/la centralité, sorties
qui ne participent pas au prompt SAM 2.

Par défaut, Wasserstein borne chaque support à 2 000 pixels par sous-échantillonnage
déterministe, afin que la matrice de coût reste calculable. Les JSON indiquent
explicitement que la mesure est approchée. `CRACKSAM2_WASSERSTEIN_EXACT=1`
active le support complet, avec un coût mémoire potentiellement prohibitif.

## Commandes séparées

Préparation des données :

```bash
python ISPRS/CrackSAM/prepare_cracksam2_data.py \
  --output /data/cracksam2 --datasets khanhha road420 facade390 concrete3k
```

Pré-calcul d'un split :

```bash
python ISPRS/CrackSAM/precompute_frangi_prompts.py \
  --data-root /data/cracksam2/khanhha/train \
  --list-file ISPRS/CrackSAM/CrackSAM/CrackSAM/lists/lists_khanhha/train.txt \
  --cache-dir /data/prompts/khanhha/train --device cuda
```

Les CLIs `train_sam2.py --help` et `evaluate_sam2.py --help` exposent les
options de smoke test (`--max-*-samples`), de reprise et de métrique.

## Artefacts

Chaque variante produit :

- `config.json`, `train.csv`, `validation.csv` ;
- `latest.pt`, checkpoint de reprise avec optimiseur ;
- `best.pt`, deltas LoRA sélectionnés sur validation ;
- `evaluation/summary.csv` et `summary.json` ;
- un `per_image.csv` et un journal reprenable `progress.jsonl` par configuration
  de test, liés par `evaluation_contract.json`.

Les checkpoints stockent l'identité SHA-256 du checkpoint SAM 2 de base afin
d'empêcher un chargement silencieux avec SAM 2.1 ou une autre taille Hiera. Le
checkpoint de reprise embarque aussi un contrat immuable couvrant données,
caches, LoRA, perte, optimiseur, planning, augmentation et graine.

Les listes CrackSAM utilisent 9 602 des 9 603 images Khanhha d'entraînement :
`CFD_042.jpg`, présent dans l'archive, n'est référencé ni par `train.txt` ni par
`val_vol.txt` et reste donc volontairement hors protocole.
