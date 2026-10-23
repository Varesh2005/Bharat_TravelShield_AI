"""
Bharat-TravelShield AI - Spatiotemporal Multi-Hazard ANN Training Pipeline
Author: Varesh (ML / ANN & Research) + Pranav
Project: Bharat-TravelShield AI

Trains the MultiHazardRiskANN multi-task architecture on India-wide
spatiotemporal climate, epidemiology, and disaster data.
Computes ROC-AUC, PR-AUC, and Brier Score per hazard.
Exports weights to both PyTorch .pt and lightweight portable JSON.
"""

import os
import json
import random
from typing import Dict, Tuple, Any
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from ml.models.ann_model import MultiHazardRiskANN, FEATURE_NAMES, NUM_FEATURES


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def generate_spatiotemporal_dataset(num_samples: int = 12000, seed: int = 42) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Generates realistic, physically grounded synthetic spatiotemporal records for Indian districts
    matching real-world IMD, NVBDCP, and NDMA patterns:
      - Monsoon months (July-Sept) + High Humidity + High Rainfall -> High Dengue & Flood risk
      - Rural/forest regions + Post-monsoon + Warm Temp -> High Malaria risk
      - Summer months (April-June) + High Max Temp (>40°C) + Low Humidity -> High Heatwave risk
      - High altitude / Winter -> Low Heatwave, Low Dengue, Low Flood
    """
    np.random.seed(seed)

    # 1. Spatiotemporal coordinates across India
    # Lat: 8°N to 35°N, Lon: 68°E to 96°E
    lat = np.random.uniform(8.5, 34.0, num_samples)
    lon = np.random.uniform(69.0, 95.0, num_samples)
    elevation = np.random.exponential(300.0, num_samples)
    # High Himalayas in north
    elevation = np.where(lat > 30.0, elevation + np.random.uniform(1000.0, 3500.0, num_samples), elevation)

    # Month: 1 to 12
    month = np.random.randint(1, 13, num_samples)
    month_rad = 2.0 * np.pi * (month - 1) / 12.0
    month_sin = np.sin(month_rad)
    month_cos = np.cos(month_rad)

    # Climate features based on month and latitude
    is_monsoon_bool = np.isin(month, [6, 7, 8, 9])
    is_summer_bool = np.isin(month, [4, 5, 6])
    is_winter_bool = np.isin(month, [12, 1, 2])

    is_monsoon = is_monsoon_bool.astype(np.float32)
    is_summer = is_summer_bool.astype(np.float32)
    is_winter = is_winter_bool.astype(np.float32)

    # Max & Min Temp
    base_max = np.where(is_summer_bool, 41.0, np.where(is_monsoon_bool, 31.0, np.where(is_winter_bool, 21.0, 32.0)))
    base_min = np.where(is_summer_bool, 27.0, np.where(is_monsoon_bool, 24.0, np.where(is_winter_bool, 9.0, 20.0)))
    # High altitude cooling (-6.5°C per 1000m lapse rate)
    lapse = (elevation / 1000.0) * 6.5
    max_temp = base_max - lapse + np.random.normal(0, 2.5, num_samples)
    min_temp = base_min - lapse + np.random.normal(0, 2.0, num_samples)
    diurnal_range = np.clip(max_temp - min_temp, 1.0, 25.0)

    # Rainfall
    base_r7 = np.where(is_monsoon_bool, np.random.exponential(90.0, num_samples), np.random.exponential(8.0, num_samples))
    base_r30 = base_r7 * np.random.uniform(2.5, 4.5, num_samples)
    # Mountain rain shadow or western ghats heavy rain
    is_western_ghats = (lon < 76.0) & (lat < 18.0) & (lat > 9.0)
    base_r7 = np.where(is_western_ghats & is_monsoon_bool, base_r7 * 1.8, base_r7)
    base_r30 = np.where(is_western_ghats & is_monsoon_bool, base_r30 * 1.8, base_r30)

    # Humidity
    humidity = np.where(is_monsoon_bool, np.random.uniform(75.0, 95.0, num_samples),
                        np.where(is_summer_bool, np.random.uniform(20.0, 50.0, num_samples),
                                 np.random.uniform(45.0, 75.0, num_samples)))

    # Disease priors
    pop_density = np.random.exponential(900.0, num_samples) + 50.0
    dengue_cases = np.where(is_monsoon_bool & (pop_density > 500.0), np.random.exponential(120.0, num_samples), np.random.exponential(15.0, num_samples))
    malaria_cases = np.where(is_monsoon_bool & (elevation < 1200.0), np.random.exponential(60.0, num_samples), np.random.exponential(10.0, num_samples))

    # Historical frequencies
    flood_freq = np.clip((base_r30 / 400.0) * 0.7 + np.random.uniform(0.0, 0.3, num_samples), 0.0, 1.0)
    heatwave_freq = np.clip(is_summer * (max_temp / 45.0) * 0.8 + np.random.uniform(0.0, 0.2, num_samples), 0.0, 1.0)

    # Normalized Features
    lat_norm = (lat - 20.0) / 10.0
    lon_norm = (lon - 80.0) / 10.0
    elev_norm = elevation / 1000.0
    max_t_norm = (max_temp - 30.0) / 10.0
    min_t_norm = (min_temp - 20.0) / 10.0
    diurnal_norm = diurnal_range / 10.0
    r7_norm = base_r7 / 100.0
    r30_norm = base_r30 / 300.0
    rh_norm = humidity / 100.0
    summer_heat_flag = ((is_summer == 1.0) & (max_temp >= 38.0)).astype(np.float32)
    dengue_lag_norm = np.log1p(dengue_cases) / 5.0
    malaria_lag_norm = np.log1p(malaria_cases) / 5.0
    pop_density_norm = np.log1p(pop_density) / 10.0

    X = np.column_stack([
        month_sin,
        month_cos,
        lat_norm,
        lon_norm,
        elev_norm,
        max_t_norm,
        min_t_norm,
        diurnal_norm,
        r7_norm,
        r30_norm,
        rh_norm,
        is_monsoon,
        summer_heat_flag,
        dengue_lag_norm,
        malaria_lag_norm,
        flood_freq,
        heatwave_freq,
        pop_density_norm,
    ]).astype(np.float32)

    # 2. Physics-informed Ground Truth Probabilities & Binary Labels
    # Dengue: Mosquito breeding favored by stagnant water (rainfall), warm humid weather (24-33°C), urban density
    dengue_logit = (
        1.8 * is_monsoon +
        1.2 * rh_norm +
        0.8 * r7_norm +
        1.0 * dengue_lag_norm +
        0.7 * pop_density_norm -
        2.5 * elev_norm -
        1.5 * is_winter - 2.8
    )
    p_dengue = 1.0 / (1.0 + np.exp(-dengue_logit))
    y_dengue = (np.random.uniform(0, 1, num_samples) < p_dengue).astype(np.float32)

    # Malaria: Anopheles breeding in rural/fringe areas with standing water, temperature 20-30°C
    malaria_logit = (
        1.5 * is_monsoon +
        0.9 * rh_norm +
        1.2 * malaria_lag_norm +
        0.6 * r30_norm -
        2.2 * elev_norm -
        1.0 * pop_density_norm - 2.5
    )
    p_malaria = 1.0 / (1.0 + np.exp(-malaria_logit))
    y_malaria = (np.random.uniform(0, 1, num_samples) < p_malaria).astype(np.float32)

    # Flood: Heavy 7-day and 30-day cumulative rainfall exceeding drainage capacity
    flood_logit = (
        2.2 * r7_norm +
        1.8 * r30_norm +
        1.5 * flood_freq +
        1.0 * is_monsoon -
        1.8 * elev_norm - 3.2
    )
    p_flood = 1.0 / (1.0 + np.exp(-flood_logit))
    y_flood = (np.random.uniform(0, 1, num_samples) < p_flood).astype(np.float32)

    # Heatwave: Extreme max temp (>40°C), summer season, low humidity, high diurnal range
    heatwave_logit = (
        3.0 * summer_heat_flag +
        2.2 * max_t_norm +
        1.5 * heatwave_freq +
        1.0 * diurnal_norm -
        2.5 * rh_norm -
        1.8 * elev_norm - 2.8
    )
    p_heatwave = 1.0 / (1.0 + np.exp(-heatwave_logit))
    y_heatwave = (np.random.uniform(0, 1, num_samples) < p_heatwave).astype(np.float32)

    y_dict = {
        "dengue": y_dengue,
        "malaria": y_malaria,
        "flood": y_flood,
        "heatwave": y_heatwave,
    }

    return X, y_dict


def train_model(
    epochs: int = 25,
    batch_size: int = 64,
    learning_rate: float = 0.003,
    output_dir: str = "ml/models/weights",
) -> Dict[str, Any]:
    """
    Executes complete training loop for the MultiHazardRiskANN model.
    Saves trained checkpoints and returns validation metrics.
    """
    set_seed(42)
    os.makedirs(output_dir, exist_ok=True)

    print("=== Bharat-TravelShield AI Multi-Hazard ANN Training ===")
    print("Generating spatiotemporal dataset (12,000 samples across Indian districts)...")
    X, y_dict = generate_spatiotemporal_dataset(num_samples=12000, seed=42)

    # Train / Val Split (80% Train, 20% Validation)
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]

    y_train = {k: v[:split_idx] for k, v in y_dict.items()}
    y_val = {k: v[split_idx:] for k, v in y_dict.items()}

    print(f"Dataset split: {len(X_train)} Train | {len(X_val)} Validation")

    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for training MultiHazardRiskANN.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Prepare PyTorch Tensors
    X_tr_t = torch.tensor(X_train, dtype=torch.float32)
    y_tr_t = {k: torch.tensor(v, dtype=torch.float32).unsqueeze(1) for k, v in y_train.items()}

    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = {k: torch.tensor(v, dtype=torch.float32).unsqueeze(1) for k, v in y_val.items()}

    # Initialize Model & Optimizer
    model = MultiHazardRiskANN(input_dim=NUM_FEATURES, dropout_rate=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.BCELoss()

    # DataLoader
    train_dataset = TensorDataset(
        X_tr_t,
        y_tr_t["dengue"],
        y_tr_t["malaria"],
        y_tr_t["flood"],
        y_tr_t["heatwave"],
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Training Loop
    print(f"Beginning training for {epochs} epochs...")
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch_x, b_dengue, b_malaria, b_flood, b_heatwave in train_loader:
            batch_x = batch_x.to(device)
            targets = {
                "dengue": b_dengue.to(device),
                "malaria": b_malaria.to(device),
                "flood": b_flood.to(device),
                "heatwave": b_heatwave.to(device),
            }

            optimizer.zero_grad()
            preds = model(batch_x)

            # Multi-Task Loss: Sum of BCE losses across 4 hazards
            loss_dengue = criterion(preds["dengue"], targets["dengue"])
            loss_malaria = criterion(preds["malaria"], targets["malaria"])
            loss_flood = criterion(preds["flood"], targets["flood"])
            loss_heatwave = criterion(preds["heatwave"], targets["heatwave"])

            total_loss = loss_dengue + loss_malaria + loss_flood + loss_heatwave
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item() * batch_x.size(0)

        epoch_loss = running_loss / len(X_train)

        # Validation every 5 epochs
        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                val_preds = model(X_val_t.to(device))
                val_loss = (
                    criterion(val_preds["dengue"], y_val_t["dengue"].to(device)) +
                    criterion(val_preds["malaria"], y_val_t["malaria"].to(device)) +
                    criterion(val_preds["flood"], y_val_t["flood"].to(device)) +
                    criterion(val_preds["heatwave"], y_val_t["heatwave"].to(device))
                ).item()

            print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Final Comprehensive Evaluation
    model.eval()
    with torch.no_grad():
        val_preds_dict = model(X_val_t.to(device))

    metrics = {}
    for hazard in ["dengue", "malaria", "flood", "heatwave"]:
        y_true = y_val[hazard]
        y_prob = val_preds_dict[hazard].cpu().numpy().flatten()

        roc_auc = float(roc_auc_score(y_true, y_prob))
        pr_auc = float(average_precision_score(y_true, y_prob))
        brier = float(brier_score_loss(y_true, y_prob))

        metrics[hazard] = {
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "brier_score": round(brier, 4),
            "positive_prevalence": round(float(np.mean(y_true)), 4),
        }
        print(f"[{hazard.upper()}] ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | Brier: {brier:.4f}")

    # Save Artifacts
    pt_path = os.path.join(output_dir, "travelshield_ann.pt")
    torch.save(model.state_dict(), pt_path)
    print(f"Saved PyTorch weights: {pt_path}")

    # Save Portable JSON Weights (for pure NumPy inference anywhere)
    json_path = os.path.join(output_dir, "travelshield_weights.json")
    exported_weights = model.export_weights_to_dict()
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(exported_weights, f)
    print(f"Saved portable JSON weights: {json_path}")

    # Save Metrics File
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": "MultiHazardRiskANN",
            "version": "1.0.0",
            "num_features": NUM_FEATURES,
            "feature_names": FEATURE_NAMES,
            "metrics": metrics,
        }, f, indent=2)
    print(f"Saved evaluation metrics: {metrics_path}")

    return metrics


if __name__ == "__main__":
    train_model(epochs=25)
