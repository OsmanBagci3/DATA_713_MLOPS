# PaySim Fraud Detection — MLOps Project

Détection de fraude dans les transactions de mobile money avec un pipeline MLOps complet.

**Dataset** : [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) — 6,3M de transactions simulées de mobile money.

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Airflow     │────▶│   MinIO (S3) │◀────│    MLflow     │
│  (DAGs/CT)    │     │  (stockage)  │     │  (tracking)   │
└──────┬───────┘     └──────────────┘     └──────┬───────┘
       │                                          │
       ▼                                          ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Feature Eng. │────▶│   Training    │────▶│ Model Registry│
│  (PaySim)     │     │ (RandomForest)│     │  (MLflow)     │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                                                  ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Streamlit    │────▶│   FastAPI     │◀───│  Prometheus   │
│  (webapp)     │     │  (serving)    │     │  + Grafana    │
└──────────────┘     └──────────────┘     └──────────────┘
```

## Stack technique

| Composant | Technologie |
|-----------|-------------|
| ML Framework | Scikit-learn (RandomForest) |
| Orchestrateur | Apache Airflow |
| Tracking/Registry | MLflow |
| Stockage | MinIO (S3-compatible) |
| API | FastAPI |
| Webapp | Streamlit |
| Monitoring | Prometheus + Grafana |
| CI/CD | GitHub Actions |
| Conteneurisation | Docker / Docker Compose |
| Déploiement | Kubernetes (Minikube) |

## Particularités du dataset PaySim

Le dataset PaySim simule un mois de transactions de mobile money (744 heures). Les colonnes brutes sont :

- **step** : heure de la simulation (1 step = 1 heure)
- **type** : CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER
- **amount** : montant de la transaction
- **nameOrig/nameDest** : identifiants émetteur/destinataire
- **oldbalanceOrg/newbalanceOrig** : solde émetteur avant/après
- **oldbalanceDest/newbalanceDest** : solde destinataire avant/après
- **isFraud** : label (fraude uniquement sur TRANSFER et CASH_OUT)
- **isFlaggedFraud** : flag business pour transferts > 200k

### Feature engineering

Le preprocessing crée 15 features à partir des colonnes brutes :

1. `type_encoded` : type de transaction encodé en entier
2. `amount` : montant (scalé)
3. `oldbalanceOrg/newbalanceOrig` : soldes émetteur (scalés)
4. `oldbalanceDest/newbalanceDest` : soldes destinataire (scalés)
5. `orig_balance_diff` : variation du solde émetteur
6. `dest_balance_diff` : variation du solde destinataire
7. `orig_balance_error` : écart entre le montant et la variation réelle
8. `dest_balance_error` : idem côté destinataire
9. `amount_ratio_orig` : ratio montant / solde émetteur
10. `is_orig_empty_after` : le compte émetteur est vidé (flag binaire)
11. `is_dest_empty_before` : le compte destinataire était vide (flag binaire)
12. `step_hour` : heure de la journée (0-23)
13. `step_day` : jour de la simulation (0-30)

## Quickstart

### 1. Lancer l'environnement
```bash
make dev
```

### 2. Préparer les données
Télécharger le CSV depuis [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1) et l'uploader dans MinIO (bucket `data`) via http://localhost:9001.

### 3. Lancer les pipelines
Via Airflow (http://localhost:8080) : activer `data_pipeline` puis `retrain_pipeline`.

### 4. Tester l'API
```bash
curl -X POST http://localhost:8000/predict/raw \
  -H "Content-Type: application/json" \
  -H "X-API-Key: admin-key-123" \
  -d '{"step":1,"type":"TRANSFER","amount":181000,"oldbalanceOrg":181000,"newbalanceOrig":0,"oldbalanceDest":0,"newbalanceDest":0}'
```

## Structure
```
├── airflow/dags/           # 3 DAGs (data, retrain, live simulation)
├── src/data/               # Download + feature engineering
├── src/train/              # Training + evaluation/promotion
├── src/api/                # FastAPI (2 endpoints predict: raw et features)
├── src/webapp/             # Streamlit 4 pages
├── tests/                  # Unit, integration, e2e
├── k8s/                    # Kubernetes manifests
├── .github/workflows/      # CI/CD
├── docker-compose.yml      # Env dev (10 services)
├── locustfile.py           # Tests de charge
└── Makefile                # Commandes utiles
```

## Métriques
F1-Score (promotion), Precision, Recall, ROC-AUC, PR-AUC. L'accuracy n'est pas utilisée (dataset à 0,13% de fraude).

## Commandes
```bash
make help        # Liste des commandes
make dev         # Lancer l'env
make test        # Tests
make loadtest    # Locust
make k8s-deploy  # Déployer sur K8s
```
