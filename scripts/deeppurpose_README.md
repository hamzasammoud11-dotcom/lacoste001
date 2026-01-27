DeepPurpose002 — Training & Prediction (DTI)

Ce repo contient un pipeline DeepPurpose pour :

entraîner un modèle Drug–Target Interaction (DTI) à partir de paires (SMILES, séquence protéique, label),

évaluer le modèle (métriques + logs),

prédire des interactions/affinités sur de nouvelles paires et exporter les résultats.

Contenu

deeppurpose002.py : chargement données → preprocessing/encodage → entraînement → évaluation → sauvegarde modèle + outputs

prediction_test.py (ou équivalent) : chargement du modèle sauvegardé → prédictions → export CSV

Utilisation
python deeppurpose002.py
python prediction_test.py

Format attendu

Train (supervisé) : drug_smiles, target_sequence, label

Predict : drug_smiles, target_sequence

Outputs : modèles dans models/, résultats/logs dans outputs/.
