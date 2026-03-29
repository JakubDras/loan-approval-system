import pytorch_lightning as pl
import mlflow
import mlflow.pytorch
import torch
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.preprocess import load_config, create_dataloaders
from src.models.architecture import LoanApprovalModel
from src.models.compression import compute_neuron_importance, prune_neurons_by_xai, compress_model_physically, \
    apply_dynamic_quantization


def evaluate_model(model, loader, device="cpu"):
    model.eval()
    model.to(device)
    correct, total = 0, 0

    with torch.no_grad():
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            y_out = model(X_b)
            preds = (y_out > 0.5).float()
            correct += (preds == y_b).sum().item()
            total += y_b.size(0)

    return correct / total


def optimize():
    config = load_config()
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("Loan_Approval_Optimization")

    train_loader, val_loader, test_loader, input_dim = create_dataloaders(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Pobieranie ostatniego modelu z MLflow...")
    runs = mlflow.search_runs(experiment_names=["Loan_Approval_Optimization"], order_by=["start_time DESC"])
    last_run_id = runs.iloc[0].run_id
    model_uri = f"runs:/{last_run_id}/baseline_model"

    model = mlflow.pytorch.load_model(model_uri)
    model.to(device)

    with mlflow.start_run(run_name="Optimization_Pipeline"):
        mlflow.log_param("pruning_percentage", config['compression']['pruning_percentage'])

        print(f"\n1. Obliczanie ważności neuronów (Taylor/XAI) na: {device}...")
        scores = compute_neuron_importance(model, train_loader, device)

        print("   Nakładanie masek...")
        pruned_model = prune_neurons_by_xai(model, scores, config['compression']['pruning_percentage'])

        print("   Fizyczna kompresja (usuwanie zamaskowanych wag)...")
        compact_net = compress_model_physically(pruned_model, input_dim)

        print("\n2. Kwantyzacja do Int8 (CPU)...")
        quantized_model = apply_dynamic_quantization(compact_net)

        print("\n3. Walidacja skompresowanego modelu...")
        final_acc = evaluate_model(quantized_model, test_loader, device=torch.device("cpu"))
        mlflow.log_metric("final_test_acc", final_acc)

        temp_path = "temp_quantized.pth"
        torch.save(quantized_model.state_dict(), temp_path)
        size_kb = os.path.getsize(temp_path) / 1024
        os.remove(temp_path)
        mlflow.log_metric("final_model_size_kb", size_kb)

        mlflow.pytorch.log_model(quantized_model, "production_model")

        print(f"\n=== PODSUMOWANIE OPTYMALIZACJI ===")
        print(f"Finalny rozmiar modelu: {size_kb:.2f} KB")
        print(f"Finalne Accuracy:       {final_acc:.4f}")
        print("Model został zapisany jako 'production_model' w MLflow.")


if __name__ == "__main__":
    pl.seed_everything(42)
    optimize()