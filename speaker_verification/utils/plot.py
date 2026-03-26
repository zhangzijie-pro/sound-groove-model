import os
import matplotlib.pyplot as plt


def _has_nonempty(history: dict, key: str) -> bool:
    return key in history and history[key] is not None and len(history[key]) > 0


def _plot_if_exists(ax, history: dict, key: str, label: str = None):
    if _has_nonempty(history, key):
        ax.plot(history[key], label=(label or key))


def plot_curves(out_dir: str, history: dict):
    os.makedirs(out_dir, exist_ok=True)

    if _has_nonempty(history, "train_loss") or _has_nonempty(history, "val_loss"):
        fig, ax = plt.subplots(figsize=(8, 5))
        _plot_if_exists(ax, history, "train_loss", "train_loss")
        _plot_if_exists(ax, history, "val_loss", "val_loss")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title("Total Loss")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=150)
        plt.close(fig)

    train_loss_keys = [
        ("train_pit_loss", "train_pit"),
        ("train_act_loss", "train_act"),
        ("train_cnt_loss", "train_cnt"),
        ("train_frm_loss", "train_frm"),
    ]
    if any(_has_nonempty(history, k) for k, _ in train_loss_keys):
        fig, ax = plt.subplots(figsize=(8, 5))
        for k, label in train_loss_keys:
            _plot_if_exists(ax, history, k, label)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title("Train Loss Breakdown")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "train_loss_breakdown.png"), dpi=150)
        plt.close(fig)

    val_loss_keys = [
        ("val_pit_loss", "val_pit"),
        ("val_act_loss", "val_act"),
        ("val_cnt_loss", "val_cnt"),
        ("val_frm_loss", "val_frm"),
    ]
    if any(_has_nonempty(history, k) for k, _ in val_loss_keys):
        fig, ax = plt.subplots(figsize=(8, 5))
        for k, label in val_loss_keys:
            _plot_if_exists(ax, history, k, label)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title("Val Loss Breakdown")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "val_loss_breakdown.png"), dpi=150)
        plt.close(fig)

    metric_keys = [
        ("val_der", "val_der(%)"),
        ("val_count_acc", "val_count_acc"),
        ("val_act_f1", "val_act_f1"),
    ]
    if any(_has_nonempty(history, k) for k, _ in metric_keys):
        fig, ax = plt.subplots(figsize=(8, 5))
        for k, label in metric_keys:
            _plot_if_exists(ax, history, k, label)
        ax.set_xlabel("epoch")
        ax.set_ylabel("metric")
        ax.set_title("Validation Metrics")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "val_metrics_curve.png"), dpi=150)
        plt.close(fig)

    if _has_nonempty(history, "val_der"):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(history["val_der"], label="val_der")
        ax.set_xlabel("epoch")
        ax.set_ylabel("DER (%)")
        ax.set_title("Validation DER")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "val_der_curve.png"), dpi=150)
        plt.close(fig)