import os
import matplotlib.pyplot as plt


def _has_nonempty(history: dict, key: str) -> bool:
    return key in history and history[key] is not None and len(history[key]) > 0


def _plot_if_exists(ax, history: dict, key: str, label: str = None):
    if _has_nonempty(history, key):
        ax.plot(history[key], label=(label or key))


def _finalize_plot(ax, xlabel: str, ylabel: str, title: str):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)


def plot_group(out_dir: str, history: dict, filename: str, title: str, ylabel: str, key_label_pairs):
    if not any(_has_nonempty(history, k) for k, _ in key_label_pairs):
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    for k, label in key_label_pairs:
        _plot_if_exists(ax, history, k, label)
    _finalize_plot(ax, "epoch", ylabel, title)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, filename), dpi=150)
    plt.close(fig)


def plot_curves(out_dir: str, history: dict):
    os.makedirs(out_dir, exist_ok=True)

    # 1) total loss
    plot_group(
        out_dir,
        history,
        "loss_curve.png",
        "Total Loss",
        "loss",
        [
            ("train_loss", "train_loss"),
            ("val_loss", "val_loss"),
        ],
    )

    # 2) train loss breakdown
    plot_group(
        out_dir,
        history,
        "train_loss_breakdown.png",
        "Train Loss Breakdown",
        "loss",
        [
            ("train_pit_loss", "train_pit"),
            ("train_dice_loss", "train_dice"),
            ("train_activity_loss", "train_activity"),
            ("train_exist_loss", "train_exist"),
            ("train_consistency_loss", "train_consistency"),
            ("train_smooth_loss", "train_smooth"),
        ],
    )

    # 3) val loss breakdown
    plot_group(
        out_dir,
        history,
        "val_loss_breakdown.png",
        "Val Loss Breakdown",
        "loss",
        [
            ("val_pit_loss", "val_pit"),
            ("val_dice_loss", "val_dice"),
            ("val_activity_loss", "val_activity"),
            ("val_exist_loss", "val_exist"),
            ("val_consistency_loss", "val_consistency"),
            ("val_smooth_loss", "val_smooth"),
        ],
    )

    # 4) validation metrics overview
    plot_group(
        out_dir,
        history,
        "val_metrics_curve.png",
        "Validation Metrics",
        "metric",
        [
            # ("val_der", "val_der(%)"),
            ("val_count_acc", "val_count_acc"),
            ("val_act_prec", "val_act_prec"),
            ("val_act_rec", "val_act_rec"),
            ("val_act_f1", "val_act_f1"),
            ("val_exist_acc", "val_exist_acc"),
        ],
    )

    # 5) DER only
    plot_group(
        out_dir,
        history,
        "val_der_curve.png",
        "Validation DER",
        "DER (%)",
        [
            ("val_der", "val_der"),
        ],
    )

    # 6) count / existence accuracy
    plot_group(
        out_dir,
        history,
        "count_exist_acc_curve.png",
        "Count / Existence Accuracy",
        "accuracy",
        [
            ("val_count_acc", "val_count_acc"),
            ("val_exist_acc", "val_exist_acc"),
        ],
    )

    # 7) count MAE only
    plot_group(
        out_dir,
        history,
        "val_count_mae_curve.png",
        "Validation Count MAE",
        "mae",
        [
            ("val_count_mae", "val_count_mae"),
        ],
    )

    # 8) activity metrics only
    plot_group(
        out_dir,
        history,
        "val_activity_curve.png",
        "Validation Activity Metrics",
        "metric",
        [
            ("val_act_prec", "val_act_prec"),
            ("val_act_rec", "val_act_rec"),
            ("val_act_f1", "val_act_f1"),
        ],
    )
