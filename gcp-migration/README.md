# Gestion sûre de la VM G4 Blackwell

Ce répertoire gère la VM Compute Engine `frangi-blackwell-spot`, une
`g4-standard-48` de `europe-west4-a` dotée d'une RTX PRO 6000 Blackwell
Server Edition (96 Go de VRAM). Les scripts ne suppriment jamais l'instance,
le disque ni les paquets existants.

## Préparer le contexte GCP

Toutes les commandes ciblent explicitement le même projet. Configurez-le avant
la première utilisation ; les scripts refusent un projet actif différent.

```bash
export GCP_PROJECT_ID="devpod-gpu-exploration"
gcloud config set project "${GCP_PROJECT_ID}"
gcloud config get-value project
```

Vérifiez ensuite les quotas nécessaires :

```bash
./gcp-migration/check_quotas.sh
```

Le contrôle utilise le quota réellement disponible (`limite - usage`) et
échoue si la métrique RTX PRO 6000 Spot est absente, au lieu de supposer qu'elle
vaut un.

## Créer la VM si elle n'existe plus

```bash
./gcp-migration/deploy.sh
```

Le script demande une confirmation textuelle et refuse d'écraser une instance
existante. Ses invariants sont :

- machine `g4-standard-48` ;
- image `common-cu129-ubuntu-2204-nvidia-580` (CUDA 12.9, pilote 580) ;
- provisioning Spot avec `instanceTerminationAction=STOP` ;
- disque de démarrage Hyperdisk Balanced de 100 Go ;
- coupe-circuit GCE `maxRunDuration=8h`, réglable avant création avec
  `GCP_MAX_RUN_DURATION` (par exemple `GCP_MAX_RUN_DURATION=12h`).

L'expiration du coupe-circuit et une préemption Spot arrêtent donc la VM et
conservent son disque ; elles ne la suppriment pas.

## Ouvrir une session de calcul

Démarrez puis connectez-vous en indiquant toujours le projet :

```bash
gcloud compute instances start frangi-blackwell-spot \
  --project="${GCP_PROJECT_ID}" \
  --zone="europe-west4-a"

gcloud compute ssh frangi-blackwell-spot \
  --project="${GCP_PROJECT_ID}" \
  --zone="europe-west4-a"
```

Dans la VM, armez immédiatement un second coupe-circuit et lancez le preflight.
Le délai ci-dessous est de quatre heures et reste armé après la fin du script :

```bash
cd /chemin/vers/Generalized-Frangi-for-Automatic-Crack-Extraction-on-FIND-dataset
./gcp-migration/blackwell_preflight.sh --arm-shutdown 240
```

Sans `--arm-shutdown`, le preflight reste purement diagnostique. Son smoke test
Docker lance un conteneur `--rm` sans montage de volume ; il peut télécharger
l'image `nvidia/cuda:12.9.1-base-ubuntu22.04`. Utilisez `--skip-docker` pour un
diagnostic hors ligne.

Le feu vert exige :

- exactement une RTX PRO 6000, compute capability 12.0 ;
- au moins 95 000 MiB visibles, soit la classe nominale 96 Go ;
- un pilote de branche 580 avec module noyau ouvert ;
- `nvcc` en CUDA 12.9 ;
- un accès GPU fonctionnel depuis Docker/NVIDIA Container Toolkit.

Ne lancez les benchmarks lourds qu'après le bilan vert.

## Cas Blackwell : « requires use of the NVIDIA open kernel modules »

Le preflight cherche ce message exact dans les journaux noyau. Il affiche aussi
le module chargé et l'état des paquets. Pour une inspection manuelle :

```bash
sudo journalctl -k -b --no-pager | \
  grep -F "requires use of the NVIDIA open kernel modules"

modinfo -F license nvidia
dpkg-query -W \
  linux-modules-nvidia-580-server-open-gcp \
  nvidia-driver-580-server-open
```

Si et seulement si le message est présent, demandez la réparation :

```bash
./gcp-migration/blackwell_preflight.sh \
  --install-open-driver \
  --skip-docker
```

Le script exige la phrase `INSTALLER 580-SERVER-OPEN`, puis exécute uniquement
l'équivalent de :

```bash
sudo apt-get update
sudo apt-get install --no-remove \
  linux-modules-nvidia-580-server-open-gcp \
  nvidia-driver-580-server-open
```

Il ne lance ni `purge`, ni `autoremove`, ni reboot. Une fois l'installation
terminée, redémarrez explicitement, réarmez le coupe-circuit et exigez un
preflight entièrement vert avant les tests :

```bash
sudo reboot
# Après reconnexion :
./gcp-migration/blackwell_preflight.sh --arm-shutdown 240
```

L'option APT `--no-remove` fait échouer la réparation plutôt que de désinstaller
automatiquement un pilote en conflit. Dans ce cas, examinez le plan APT et les
paquets déjà installés avant toute intervention manuelle.

## Arrêter et vérifier après les calculs

Depuis la machine locale, utilisez le script de fermeture. Il demande de taper
`STOPPER frangi-blackwell-spot`, attend la fin de l'arrêt et n'accepte comme succès
que l'état GCE `TERMINATED` (qui signifie « arrêtée », pas « supprimée »).

```bash
./gcp-migration/stop_and_verify.sh
```

La vérification manuelle équivalente est :

```bash
gcloud compute instances stop frangi-blackwell-spot \
  --project="${GCP_PROJECT_ID}" \
  --zone="europe-west4-a"

gcloud compute instances describe frangi-blackwell-spot \
  --project="${GCP_PROJECT_ID}" \
  --zone="europe-west4-a" \
  --format='value(status)'
```

La dernière commande doit afficher `TERMINATED`. Si ce n'est pas le cas,
contrôlez immédiatement la VM dans la console GCP.
