.PHONY: help dev up down test lint train loadtest k8s-deploy

help: ## Show help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

dev: ## Start all services
	docker-compose up -d --build
	@echo "  API:        http://localhost:8000/docs"
	@echo "  Webapp:     http://localhost:8501"
	@echo "  MLflow:     http://localhost:5000"
	@echo "  Airflow:    http://localhost:8080 (admin/admin)"
	@echo "  MinIO:      http://localhost:9001 (minioadmin/minioadmin)"
	@echo "  Grafana:    http://localhost:3000 (admin/admin)"
	@echo "  Prometheus: http://localhost:9090"
	@echo "  MailHog:    http://localhost:8025"

up: dev ## Alias for dev
down: ## Stop services
	docker-compose down
down-clean: ## Stop + remove volumes
	docker-compose down -v
logs: ## Show logs
	docker-compose logs -f

test: ## Run all tests
	pytest tests/ -v
test-unit: ## Unit tests
	pytest tests/unit/ -v
test-integration: ## Integration tests
	pytest tests/integration/ -v
lint: ## Lint code
	ruff check src/ tests/ airflow/

train: ## Train locally
	MLFLOW_TRACKING_URI=http://localhost:5000 python -m src.train.train
loadtest: ## Run Locust (http://localhost:8089)
	locust -f locustfile.py --host=http://localhost:8000

k8s-deploy: ## Deploy to Kubernetes
	kubectl apply -f k8s/
k8s-status: ## K8s status
	kubectl get pods && kubectl get services
build-api: ## Build API image
	docker build -f Dockerfile.api -t fraud-api:latest .
build-webapp: ## Build webapp image
	docker build -f Dockerfile.webapp -t fraud-webapp:latest .
build-all: build-api build-webapp ## Build all images
