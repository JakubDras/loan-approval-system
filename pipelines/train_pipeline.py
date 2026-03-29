import pytorch_lightning as pl
import mlflow
import mlflow.pytorch
import torch

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.preprocess import load_config, create_dataloaders
from src.models.architecture import LoanApprovalModel


def train():
    config = load_config()

    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("Loan_Approval_Optimization")

    print("Pobieranie i przetwarzanie danych...")
    train_loader, val_loader, test_loader, input_dim = create_dataloaders(config)

    with mlflow.start_run(run_name="Baseline_Training"):
        mlflow.log_params(config['model'])
        mlflow.log_param("input_dim", input_dim)
        mlflow.log_param("pruned_features_count", len(config['data']['features_to_drop']))

        print(f"Inicjalizacja modelu z wejściem: {input_dim}")
        model = LoanApprovalModel(
            input_dim=input_dim,
            lr=config['model']['learning_rate'],
            weight_decay=config['model']['weight_decay']
        )

        trainer = pl.Trainer(
            max_epochs=config['model']['max_epochs'],
            accelerator="auto",
            enable_progress_bar=True,
            logger=False
        )

        print("Rozpoczęcie treningu...")
        trainer.fit(model, train_loader, val_loader)

        print("Ewaluacja modelu na zbiorze testowym...")
        test_results = trainer.test(model, test_loader)[0]

        mlflow.log_metric("test_acc", test_results['test_acc'])
        mlflow.log_metric("test_f1", test_results['test_f1'])

        temp_path = "temp_baseline.pth"
        torch.save(model.state_dict(), temp_path)
        size_kb = os.path.getsize(temp_path) / 1024
        mlflow.log_metric("model_size_kb", size_kb)
        os.remove(temp_path)

        mlflow.pytorch.log_model(model, "baseline_model")

        print(f"Sukces! Trening zakończony. Rozmiar modelu: {size_kb:.2f} KB")


if __name__ == "__main__":
    pl.seed_everything(42)
    train()