# VM G4 Blackwell — exécution sûre et reprenable

Ce dossier pilote des sessions Compute Engine sur `g4-standard-48` : une RTX
PRO 6000 Blackwell Server Edition de 96 Go, 48 vCPU et 180 Go de RAM. Il reprend
les gardes opérationnelles ajoutées dans
[`Ludwig-H/E-HGP`](https://github.com/Ludwig-H/E-HGP/tree/main/gcp-migration),
adaptées à CrackSAM et au projet `devpod-gpu-exploration`.

La mise à jour du 20 juillet 2026 n'a créé, démarré, modifié ni supprimé aucune
ressource GCP. Les contrôles de capacité et de quota sont en lecture seule ; les
créations et démarrages exigent une confirmation explicite.

## État et choix de zone

L'instance historique est :

```text
devpod-gpu-exploration / europe-west4-a / frangi-blackwell-spot
```

Au dernier audit elle était `TERMINATED`, en Spot avec action `STOP`, durée
maximale de huit heures et Hyperdisk Balanced de 100 Go. Elle précède les
nouveaux labels et l'activation OS Login ; les scripts l'acceptent uniquement
par ce triplet exact en mode legacy borné.

Google documente les zones G4 européennes suivantes au 20 juillet 2026 :

| Région | Zones G4 |
|---|---|
| `europe-north1` | `a`, `b`, `c` |
| `europe-west1` | `c` |
| `europe-west2` | `a`, `b`, `c` |
| `europe-west4` | `a`, `b`, `c`, `ai1a` |
| `europe-west8` | `b`, `c` |
| `europe-west10` | `b` |

Pour le projet au moment de l'audit, le conseil Spot était :

| Zone | Obtainability | Uptime estimé |
|---|---:|---:|
| `europe-west4-ai1a` | `0,9` | `3 600 s` |
| `europe-west8-c` | `0,9` | `3 600 s` |
| `europe-west4-a`, `b`, `c` | `0,1` | `60 s` |
| `europe-west8-b` | `0,1` | `60 s` |

Un score Capacity Advisor est ponctuel : il ne réserve rien et ne garantit pas
une création. L'ordre recommandé est : cible historique `europe-west4-a`, puis
nouvelle cible explicite `europe-west4-ai1a`, puis repli interrégional explicite
`europe-west8-c`. Aucun script ne change de zone automatiquement.

## Scripts et mutations

| Script | Effet |
|---|---|
| `check_capacity.sh` | lecture seule : visibilité et conseil Spot |
| `check_quotas.sh` | lecture seule : quotas Cloud Quotas exacts |
| `deploy.sh` | crée une nouvelle VM, puis l'arrête et certifie `TERMINATED` |
| `start_and_verify.sh` | démarre une VM et arme le shutdown invité |
| `blackwell_preflight.sh` | diagnostic invité ; réparation pilote seulement sur option et double confirmation |
| `stop_and_verify.sh` | arrête la cible exacte et certifie `TERMINATED` |

## Configuration locale

Copier le modèle non secret ; `.env.gcp` est ignoré par Git :

```bash
cp gcp-migration/.env.example .env.gcp
```

Les scripts ne sourcent aucun fichier implicitement. Après vérification du
contenu :

```bash
set -a
source .env.gcp
set +a

gcloud config set project "${GCP_PROJECT_ID}"
gcloud config get-value project
gcloud config get-value account
```

Pour le repli dans la zone IA, modifier ensemble :

```bash
GCP_REGION=europe-west4
GCP_ZONE=europe-west4-ai1a
GCP_INSTANCE_NAME=frangi-blackwell-spot-ai1a
```

Le projet doit déjà avoir `ai-zones-visibility=ENABLED`. `deploy.sh` le vérifie
mais ne l'active jamais à votre place.

## Préflight en lecture seule

```bash
./gcp-migration/check_capacity.sh
./gcp-migration/check_quotas.sh
```

Le quota CPU Spot est seulement informatif : G4 ne consomme pas de quota CPU.
Le contrôle bloque en revanche si le quota exact
`PREEMPTIBLE-NVIDIA-RTX-PRO-6000-GPUS-per-project-region` est absent ou épuisé.
Il vérifie aussi instances, Hyperdisk Balanced, IOPS, débit et adresse externe
si le réseau configuré en demande une.

Visibilité de la machine, quota disponible, conseil de capacité et capacité
réellement obtenue sont quatre faits distincts.

## Créer une nouvelle cible

Ne lancer cette commande que si le triplet configuré n'existe pas :

```bash
./gcp-migration/deploy.sh
```

Le script refuse d'écraser une instance et exige une phrase de confirmation. Il
crée :

- `g4-standard-48`, Spot, action `STOP` ;
- `maintenance-policy=TERMINATE`, sans redémarrage automatique ;
- `maxRunDuration` entre 30 secondes et 8 heures ;
- image DLVM CUDA 12.9 / pilote 580 résolue par nom exact ;
- Hyperdisk Balanced 100 Go, 3 600 IOPS et 290 MiB/s ;
- gVNIC avec adresse externe par défaut ;
- OS Login, labels CrackSAM et deletion protection ;
- aucun service account par défaut.

Une identité de runtime n'est ajoutée que si
`GCP_RUNTIME_SERVICE_ACCOUNT` est explicitement défini. Après création, la VM
est immédiatement arrêtée et doit être certifiée `TERMINATED`.

Créer dans une autre zone signifie créer un nouveau disque facturé et organiser
explicitement la copie des données. Ne réutilisez pas le nom de la cible
historique.

## Démarrer une session

```bash
./gcp-migration/start_and_verify.sh
```

Le script :

1. vérifie le type, Spot, `STOP`, la maintenance, le redémarrage et la durée ;
2. demande confirmation ;
3. démarre la cible et enregistre son `lastStartTimestamp` ;
4. arme un shutdown invité, puis relit sa preuve ;
5. tente un arrêt borné à cette génération si la certification échoue.

Une fois connecté à la VM, depuis le dépôt :

```bash
./gcp-migration/blackwell_preflight.sh
```

Il vérifie une RTX PRO 6000, CC 12.0, au moins 95 000 MiB visibles, pilote 580
à module noyau ouvert, `nvcc` 12.9 et Docker/NVIDIA Container Toolkit. Son
conteneur de test est éphémère et sans volume ; `--skip-docker` permet un
diagnostic hors ligne.

Pour une session démarrée hors du point d'entrée certifié, armer une garde
invitée d'au plus 480 minutes :

```bash
./gcp-migration/blackwell_preflight.sh --arm-shutdown 240
```

### Réparation exceptionnelle du pilote ouvert

Seulement si le noyau contient le message exact
`requires use of the NVIDIA open kernel modules` :

```bash
./gcp-migration/blackwell_preflight.sh \
  --install-open-driver \
  --skip-docker
```

Le script exige `INSTALLER 580-SERVER-OPEN`, installe uniquement les paquets
ouverts avec APT `--no-remove`, ne purge rien et ne redémarre pas. Le reboot et
le nouveau preflight restent explicites.

## Arrêter et certifier

Après synchronisation des checkpoints et rapports :

```bash
./gcp-migration/stop_and_verify.sh
```

Le succès exige `TERMINATED`. Le script vérifie l'identité de la cible, peut
verrouiller l'arrêt sur `lastStartTimestamp`, et affiche les autres G4 sans les
modifier. Le quota global disponible ne permettant qu'une G4 concurrente, ne
faites pas chevaucher les campagnes CrackSAM et E-HGP.

## Sources

- [Zones GPU Compute Engine](https://cloud.google.com/compute/docs/gpus/gpu-regions-zones)
- [Quotas Compute Engine](https://cloud.google.com/compute/resource-usage)
- [Capacity Advisor](https://cloud.google.com/compute/docs/instances/view-vm-availability)
- [Images Deep Learning VM](https://cloud.google.com/deep-learning-vm/docs/images)
- Mises à niveau E-HGP : [`601334c`](https://github.com/Ludwig-H/E-HGP/commit/601334ce), [`0fbdcaf`](https://github.com/Ludwig-H/E-HGP/commit/0fbdcaf2), [`e06e573`](https://github.com/Ludwig-H/E-HGP/commit/e06e573c)

Les contrats statiques et la syntaxe Bash se vérifient hors connexion avec :

```bash
pytest -q gcp-migration/tests
```
