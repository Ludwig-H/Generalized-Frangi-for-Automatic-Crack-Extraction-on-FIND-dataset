# Exécuter la feuille de route sur une VM G4

La feuille de route cible une `g4-standard-48` Spot : une RTX PRO 6000
Blackwell Server Edition de 96 Go, 48 vCPU et 180 Go de RAM. Les scripts
opérationnels sont centralisés dans [`gcp-migration/`](../../../gcp-migration/README.md).

## Principe de sécurité

Une session GPU est une opération facturable et explicitement autorisée. Les
outils du dépôt séparent donc quatre questions :

1. le type G4 est-il visible dans la zone ?
2. les quotas exacts sont-ils disponibles ?
3. Capacity Advisor estime-t-il une capacité Spot plausible ?
4. la création ou le démarrage demandé a-t-il effectivement réussi ?

Les trois premiers contrôles sont en lecture seule. Aucun script ne bascule
automatiquement dans une autre zone, car cela créerait un nouveau disque et une
nouvelle ressource facturée.

## Cibles recommandées au 20 juillet 2026

| Priorité | Zone | Usage |
|---:|---|---|
| 1 | `europe-west4-a` | instance historique existante, arrêtée |
| 2 | `europe-west4-ai1a` | premier repli explicite dans la même région |
| 3 | `europe-west8-c` | repli interrégional explicite |

Lors du dernier relevé, Capacity Advisor donnait `0,9` d'obtainability à
`europe-west4-ai1a` et `europe-west8-c`, contre `0,1` à
`europe-west4-a`. Ce relevé est ponctuel, ne réserve aucune capacité et doit
être régénéré avant une création.

## Avant chaque session

Depuis la racine du dépôt :

```bash
cp gcp-migration/.env.example .env.gcp
# Modifier la cible si nécessaire, puis charger les variables :
set -a
source .env.gcp
set +a

./gcp-migration/check_capacity.sh
./gcp-migration/check_quotas.sh
./gcp-migration/start_and_verify.sh
```

`start_and_verify.sh` vérifie les invariants Spot, démarre la génération exacte,
arme un shutdown dans l'invité et l'arrête en urgence si cette garde ne peut pas
être certifiée. Dans la VM :

```bash
./gcp-migration/blackwell_preflight.sh
```

Le démarrage a déjà armé la garde invitée ; le preflight vérifie GPU, pilote
ouvert, CUDA 12.9 et accès Docker. Pour une session lancée autrement, utiliser
`--arm-shutdown MINUTES` avec une valeur maximale de 480.

## Placement des données et reprises Spot

- code et petits manifestes : dépôt Git ;
- données, caches et checkpoints : Hyperdisk, hors du dépôt ;
- résultats légers : `ISPRS/CrackSAM/results/` après validation ;
- cache v2 : écrit par échantillon, puis manifeste publié atomiquement ;
- checkpoints : fréquents et restaurables sans recalcul complet.

Chaque session de la [roadmap](04_IMPLEMENTATION_ROADMAP.md) doit être assez
courte pour tenir sous le coupe-circuit GCE de huit heures. Une préemption Spot
doit être traitée comme un cas normal de reprise.

## Fin de session

Synchroniser les sorties durables, puis depuis la machine locale :

```bash
./gcp-migration/stop_and_verify.sh
```

La session n'est terminée que lorsque le script certifie l'état
`TERMINATED`. Le quota global du projet n'autorisant qu'une G4 concurrente, les
campagnes CrackSAM et E-HGP doivent être planifiées sans chevauchement.

## Sources opérationnelles

- [Disponibilité régionale des GPU Compute Engine](https://cloud.google.com/compute/docs/gpus/gpu-regions-zones)
- [Quotas Compute Engine](https://cloud.google.com/compute/resource-usage)
- [Capacity Advisor](https://cloud.google.com/compute/docs/instances/view-vm-availability)
- [Scripts homologues E-HGP](https://github.com/Ludwig-H/E-HGP/tree/main/gcp-migration)
