# Instructions pour l'utilisation des VMs Google Cloud (SPOT)

Ce projet utilise des instances de calcul temporaires (VMs SPOT) sur Google Cloud Platform pour l'entraînement et l'exécution des modèles tout en minimisant les coûts. Pour éviter les surcoûts et la perte de données, appliquez rigoureusement les règles suivantes :

## 1. Limitation de Durée & Arrêt de la VM
- **Coupe-circuit de sécurité** : À chaque ouverture d'une VM, assurez-vous d'avoir configuré une durée maximale d'utilisation (ex. : 4 heures).
  - Utilisez l'option `--arm-shutdown <MINUTES>` (ex. : `240` pour 4 heures) lors du lancement de `blackwell_preflight.sh` dans la VM.
- **Arrêt systématique** : N'oubliez jamais d'éteindre et de vérifier la VM à la fin de vos calculs en exécutant `./gcp-migration/stop_and_verify.sh` localement (ou via la console GCP). L'état final doit être validé à `TERMINATED`.

## 2. Entraînement Résilient (Sauvegardes fréquentes)
- **Instabilité du mode SPOT** : Les VMs SPOT peuvent être préemptées (arrêtées brusquement) par Google à tout moment.
- **Sauvegarde et Checkpoints** : Lors de l'entraînement d'un modèle, configurez impérativement des sauvegardes régulières et fréquentes (checkpoints) de l'état du modèle et de l'optimiseur.
- **Reprise des calculs** : Le code d'entraînement doit être capable de détecter la présence d'un checkpoint existant et de reprendre automatiquement les calculs là où ils s'étaient arrêtés avant la préemption.
