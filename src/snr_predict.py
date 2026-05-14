"""
SNR prediction: training, inference, saving/loading predictions, and evaluation.
"""

import os
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def make_gaussian_soft_label_matrix(num_classes, sigma=1.0):
    """Build (C, C) soft-label matrix where row i is a Gaussian distribution
    centered at class i with std sigma (in units of class index step).

    Used to implement a distance-aware loss for ordinal SNR classification:
    neighbor classes receive nonzero target probability, so misclassifying
    -20 dB as -18 dB is penalized much less than -20 dB as +18 dB.
    See guide/update1.md (method 1).
    """
    idx = np.arange(num_classes)
    diff = idx[:, None] - idx[None, :]  # row i - col j
    logits = -(diff.astype(np.float64) ** 2) / (2.0 * float(sigma) ** 2)
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    q = exp / exp.sum(axis=1, keepdims=True)
    return q.astype(np.float32)


class GaussianSoftLabelLoss(nn.Module):
    """Distance-aware cross-entropy: targets are Gaussian-smoothed one-hots."""

    def __init__(self, num_classes, sigma=1.0):
        super().__init__()
        mat = make_gaussian_soft_label_matrix(num_classes, sigma=sigma)
        self.register_buffer("soft_labels", torch.from_numpy(mat))

    def forward(self, logits, target_idx):
        log_probs = F.log_softmax(logits, dim=1)
        target = self.soft_labels[target_idx]
        return -(target * log_probs).sum(dim=1).mean()
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def train_snr_predictor(model, X_train, snr_labels_train, X_val, snr_labels_val,
                        model_path, batch_size=256, epochs=100, learning_rate=1e-3,
                        patience_lr=5, patience_es=20, factor=0.5,
                        soft_label_sigma=1.0):
    """Train SNR classifier.

    Labels (snr_labels_train / snr_labels_val) are integer class indices,
    NOT one-hot encoded and NOT raw dB values.

    Saves best and last model weights every epoch.

    Returns:
        history dict with keys: loss, accuracy, val_loss, val_accuracy, lr
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    X_train_t = torch.from_numpy(np.asarray(X_train)).float()
    X_val_t = torch.from_numpy(np.asarray(X_val)).float()
    y_train_t = torch.from_numpy(np.asarray(snr_labels_train)).long()
    y_val_t = torch.from_numpy(np.asarray(snr_labels_val)).long()

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t),
                            batch_size=batch_size, shuffle=False)

    # Distance-aware loss (Gaussian soft labels). Falls back to hard CE
    # when soft_label_sigma <= 0.
    num_snr_classes = int(max(y_train_t.max().item(), y_val_t.max().item())) + 1
    if soft_label_sigma and soft_label_sigma > 0:
        criterion = GaussianSoftLabelLoss(num_snr_classes, sigma=soft_label_sigma).to(device)
        print(f"Using GaussianSoftLabelLoss (num_classes={num_snr_classes}, sigma={soft_label_sigma})")
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=factor, patience=patience_lr, min_lr=1e-7
    )

    best_val_acc = -1.0
    best_state = None
    epochs_without_improvement = 0
    last_model_path = model_path.replace(".pt", "_last.pt")

    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": [], "lr": []}

    print(f"Training SNR predictor on {device}, saving best to {model_path}")
    start_time = time.time()

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == yb).sum().item()
            train_total += xb.size(0)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == yb).sum().item()
                val_total += xb.size(0)

        epoch_train_loss = train_loss / max(1, train_total)
        epoch_train_acc = train_correct / max(1, train_total)
        epoch_val_loss = val_loss / max(1, val_total)
        epoch_val_acc = val_correct / max(1, val_total)

        current_lr = optimizer.param_groups[0]["lr"]
        history["loss"].append(epoch_train_loss)
        history["accuracy"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_accuracy"].append(epoch_val_acc)
        history["lr"].append(current_lr)

        # LR scheduler
        old_lr = current_lr
        scheduler.step(epoch_val_acc)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < old_lr:
            print(f"ReduceLROnPlateau: lr {old_lr:.8f} -> {new_lr:.8f}")

        # Save last model every epoch
        torch.save(model.state_dict(), last_model_path)

        # Save best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, model_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"loss: {epoch_train_loss:.4f} - accuracy: {epoch_train_acc:.4f} - "
            f"val_loss: {epoch_val_loss:.4f} - val_accuracy: {epoch_val_acc:.4f} - "
            f"lr: {new_lr:.8f}"
        )

        # Early stopping
        if epochs_without_improvement >= patience_es:
            print(f"EarlyStopping at epoch {epoch + 1}. "
                  f"No val_accuracy improvement for {patience_es} epochs.")
            break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    training_time = time.time() - start_time
    print(f"SNR predictor training completed in {training_time:.2f}s, "
          f"best val_accuracy: {best_val_acc:.4f}")
    return history


def plot_snr_training_history(history, output_path):
    """Plot SNR predictor training curves (accuracy + loss)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history['accuracy'], label='Train Accuracy')
    axes[0].plot(history['val_accuracy'], label='Val Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['loss'], label='Train Loss')
    axes[1].plot(history['val_loss'], label='Val Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"SNR training history plot saved to {output_path}")


def predict_snr(model, X_data, snr_classes, batch_size=256, device=None):
    """Predict discrete SNR dB values.

    Args:
        model: Trained SNR predictor model (nn.Module)
        X_data: ndarray (N, 2, seq_len)
        snr_classes: sorted list/array of discrete SNR dB values
        batch_size: inference batch size
        device: torch device (auto-detected if None)

    Returns:
        pred_db: ndarray (N,) of predicted SNR dB values
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    snr_classes = np.asarray(snr_classes)
    X_t = torch.from_numpy(np.asarray(X_data)).float()
    loader = DataLoader(TensorDataset(X_t), batch_size=batch_size, shuffle=False)

    all_preds = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            pred_idx = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.append(pred_idx)

    pred_idx = np.concatenate(all_preds)
    pred_db = snr_classes[pred_idx]
    return pred_db


def save_snr_predictions(pred_val, pred_test, true_val, true_test, save_path):
    """Save SNR predictions to .npz file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path,
             snr_pred_val=pred_val, snr_pred_test=pred_test,
             snr_true_val=true_val, snr_true_test=true_test)
    print(f"SNR predictions saved to {save_path}")


def load_snr_predictions(load_path):
    """Load SNR predictions from .npz file.

    Returns:
        dict with keys: snr_pred_val, snr_pred_test, snr_true_val, snr_true_test
    """
    data = np.load(load_path)
    return {k: data[k] for k in data.files}


def evaluate_snr_predictor(snr_true, snr_pred, snr_classes, output_dir, split_name="val"):
    """Evaluate SNR predictor performance.

    Generates: overall accuracy, per-SNR accuracy bar chart, confusion matrix, MAE.

    Args:
        snr_true: ndarray (N,) true SNR dB values
        snr_pred: ndarray (N,) predicted SNR dB values
        snr_classes: sorted array of all discrete SNR values
        output_dir: directory to save plots and summary
        split_name: 'val' or 'test' for labeling

    Returns:
        dict with overall_accuracy, mae_db, per_snr_accuracy
    """
    from sklearn.metrics import confusion_matrix

    os.makedirs(output_dir, exist_ok=True)

    snr_classes = np.asarray(snr_classes)

    # Overall accuracy
    overall_acc = np.mean(snr_true == snr_pred)
    # MAE in dB
    mae_db = np.mean(np.abs(snr_true.astype(float) - snr_pred.astype(float)))

    print(f"SNR Predictor ({split_name}) - accuracy: {overall_acc:.4f}, MAE: {mae_db:.2f} dB")

    # Per-SNR accuracy
    per_snr_acc = {}
    for snr_val in snr_classes:
        mask = snr_true == snr_val
        if mask.sum() > 0:
            per_snr_acc[float(snr_val)] = float(np.mean(snr_pred[mask] == snr_val))

    # Plot per-SNR accuracy
    fig, ax = plt.subplots(figsize=(10, 5))
    snrs_sorted = sorted(per_snr_acc.keys())
    accs = [per_snr_acc[s] for s in snrs_sorted]
    ax.bar(range(len(snrs_sorted)), accs,
           tick_label=[str(int(s)) for s in snrs_sorted])
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"snr_per_snr_accuracy_{split_name}.png"), dpi=150)
    plt.close(fig)

    # Confusion matrix
    cm = confusion_matrix(snr_true, snr_pred, labels=snr_classes)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    tick_labels = [str(int(s)) for s in snr_classes]
    ax.set(xticks=np.arange(len(snr_classes)),
           yticks=np.arange(len(snr_classes)),
           xticklabels=tick_labels,
           yticklabels=tick_labels,
           xlabel='Predicted SNR (dB)',
           ylabel='True SNR (dB)')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"snr_confusion_matrix_{split_name}.png"), dpi=150)
    plt.close(fig)

    # Save summary text
    summary_path = os.path.join(output_dir, f"snr_eval_summary_{split_name}.txt")
    with open(summary_path, 'w') as f:
        f.write(f"SNR Predictor Evaluation ({split_name})\n")
        f.write(f"{'='*50}\n")
        f.write(f"Overall accuracy: {overall_acc:.4f}\n")
        f.write(f"MAE: {mae_db:.2f} dB\n\n")
        f.write(f"Per-SNR accuracy:\n")
        for snr_val in snrs_sorted:
            f.write(f"  {int(snr_val):>4d} dB: {per_snr_acc[snr_val]:.4f}\n")

    print(f"SNR evaluation results saved to {output_dir}")
    return {"overall_accuracy": overall_acc, "mae_db": mae_db, "per_snr_accuracy": per_snr_acc}
