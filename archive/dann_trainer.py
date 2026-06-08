import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Function
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import hybrid_trainer
import mixed_trainer
import ucddb_runner


RANDOM_STATE = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class GradientReverse(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x, alpha):
    return GradientReverse.apply(x, alpha)


class DomainECGDataset(Dataset):
    def __init__(self, x, y, domain, augment=False, aug_args=None, seed=RANDOM_STATE):
        self.x = x.astype(np.float32, copy=False)
        self.y = y.astype(np.int64, copy=False)
        self.domain = domain.astype(np.int64, copy=False)
        self.augment = augment
        self.aug_args = aug_args
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.x[idx]
        if self.augment:
            x = mixed_trainer.apply_ecg_augmentation(x, self.rng, self.aug_args)
        return (
            torch.from_numpy(x).unsqueeze(0),
            torch.tensor(self.y[idx], dtype=torch.long),
            torch.tensor(self.domain[idx], dtype=torch.long),
        )


class DANNModel(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.backbone = hybrid_trainer.ECGBackbone()
        self.label_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x, alpha=0.0):
        features = self.backbone(x)
        label_logits = self.label_classifier(features)
        domain_logits = self.domain_classifier(grad_reverse(features, alpha))
        return label_logits, domain_logits


def predict_scores(model, x, y, batch_size, device):
    domain = np.zeros(len(y), dtype=np.int64)
    loader = DataLoader(DomainECGDataset(x, y, domain, augment=False), batch_size=batch_size, shuffle=False)
    model.eval()
    labels = []
    scores = []
    with torch.no_grad():
        for batch_x, batch_y, _ in loader:
            label_logits, _ = model(batch_x.to(device), alpha=0.0)
            probs = torch.softmax(label_logits, dim=1)[:, 1]
            labels.extend(batch_y.numpy())
            scores.extend(probs.cpu().numpy())
    return np.asarray(labels, dtype=np.int64), np.asarray(scores, dtype=np.float32)


def load_splits(args):
    mixed_trainer.SHOW_PROGRESS = not args.no_progress
    ucddb_data_dir = Path(args.ucddb_dir)
    apnea_data_dir = Path(args.apnea_dir)

    ucddb_records = ucddb_runner.available_record_ids(ucddb_data_dir)
    apnea_records = mixed_trainer.available_apnea_records(apnea_data_dir)
    ucddb_train_records, ucddb_val_records, ucddb_test_records = mixed_trainer.split_records(
        ucddb_records, args.val_size, args.test_size, args.seed
    )
    apnea_train_records, apnea_val_records, apnea_test_records = mixed_trainer.split_records(
        apnea_records, args.val_size, args.test_size, args.seed
    )

    x_u_train, y_u_train, u_train_stats = mixed_trainer.load_ucddb_records(
        ucddb_data_dir,
        Path(args.ucddb_cache_dir),
        ucddb_train_records,
        args.channels,
        include_hypopnea=not args.apnea_only,
        min_overlap_sec=args.min_overlap_sec,
        context_minutes=args.context_minutes,
    )
    x_u_val, y_u_val, u_val_stats = mixed_trainer.load_ucddb_records(
        ucddb_data_dir,
        Path(args.ucddb_cache_dir),
        ucddb_val_records,
        args.channels,
        include_hypopnea=not args.apnea_only,
        min_overlap_sec=args.min_overlap_sec,
        context_minutes=args.context_minutes,
    )
    x_u_test, y_u_test, u_test_stats = mixed_trainer.load_ucddb_records(
        ucddb_data_dir,
        Path(args.ucddb_cache_dir),
        ucddb_test_records,
        args.channels,
        include_hypopnea=not args.apnea_only,
        min_overlap_sec=args.min_overlap_sec,
        context_minutes=args.context_minutes,
    )
    x_a_train, y_a_train, a_train_stats = mixed_trainer.load_apnea_records(
        apnea_data_dir, Path(args.apnea_cache_dir), apnea_train_records, args.context_minutes
    )
    x_a_val, y_a_val, a_val_stats = mixed_trainer.load_apnea_records(
        apnea_data_dir, Path(args.apnea_cache_dir), apnea_val_records, args.context_minutes
    )
    x_a_test, y_a_test, a_test_stats = mixed_trainer.load_apnea_records(
        apnea_data_dir, Path(args.apnea_cache_dir), apnea_test_records, args.context_minutes
    )

    arrays = {
        "u_train": (x_u_train, y_u_train),
        "u_val": (x_u_val, y_u_val),
        "u_test": (x_u_test, y_u_test),
        "a_train": (x_a_train, y_a_train),
        "a_val": (x_a_val, y_a_val),
        "a_test": (x_a_test, y_a_test),
    }
    records = {
        "ucddb_train": ucddb_train_records,
        "ucddb_val": ucddb_val_records,
        "ucddb_test": ucddb_test_records,
        "apnea_train": apnea_train_records,
        "apnea_val": apnea_val_records,
        "apnea_test": apnea_test_records,
    }
    record_stats = {
        "ucddb_train": u_train_stats,
        "ucddb_val": u_val_stats,
        "ucddb_test": u_test_stats,
        "apnea_train": a_train_stats,
        "apnea_val": a_val_stats,
        "apnea_test": a_test_stats,
    }
    return arrays, records, record_stats


def train(args):
    set_seed(args.seed)
    if args.context_minutes < 1 or args.context_minutes % 2 == 0:
        raise ValueError("--context-minutes must be a positive odd number.")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays, records, record_stats = load_splits(args)
    x_u_train, y_u_train = arrays["u_train"]
    x_u_val, y_u_val = arrays["u_val"]
    x_u_test, y_u_test = arrays["u_test"]
    x_a_train, y_a_train = arrays["a_train"]
    x_a_val, y_a_val = arrays["a_val"]
    x_a_test, y_a_test = arrays["a_test"]

    x_train = np.concatenate([x_u_train, x_a_train], axis=0)
    y_train = np.concatenate([y_u_train, y_a_train], axis=0)
    domain_train = mixed_trainer.source_labels(len(y_u_train), len(y_a_train))
    sampler = mixed_trainer.make_sampler(
        y_train,
        domain_train,
        args.ucddb_sample_weight,
        args.apnea_sample_weight,
    )
    train_loader = DataLoader(
        DomainECGDataset(x_train, y_train, domain_train, augment=True, aug_args=args, seed=args.seed),
        batch_size=args.batch_size,
        sampler=sampler,
    )

    model = DANNModel(dropout=args.dropout).to(device)
    label_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    domain_criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.3, patience=3)

    print(f"Using device: {device}")
    print("Normalization: ECG per-window z-score; augmentation output normalized again.")
    print(f"Context: {args.context_minutes} minute(s); DANN max lambda={args.domain_lambda}")
    for summary in [
        mixed_trainer.dataset_summary("ucddb_train", y_u_train),
        mixed_trainer.dataset_summary("ucddb_val", y_u_val),
        mixed_trainer.dataset_summary("ucddb_test", y_u_test),
        mixed_trainer.dataset_summary("apnea_train", y_a_train),
        mixed_trainer.dataset_summary("apnea_val", y_a_val),
        mixed_trainer.dataset_summary("apnea_test", y_a_test),
        mixed_trainer.dataset_summary("combined_train", y_train),
    ]:
        print(
            f"  {summary['name']}: n={summary['samples']} pos={summary['positive']} "
            f"normal={summary['normal']} pos_ratio={summary['positive_ratio']:.3f}"
        )

    best_score = -1.0
    best_threshold = 0.5
    best_ucddb_threshold = 0.5
    best_epoch = 0
    patience = 0
    total_steps = max(1, args.epochs * len(train_loader))
    global_step = 0
    model_path = output_dir / args.model_name

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_label_loss = 0.0
        running_domain_loss = 0.0
        running_domain_correct = 0
        running_seen = 0

        for batch_x, batch_y, batch_domain in tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{args.epochs}",
            leave=False,
            disable=args.no_progress,
        ):
            progress = global_step / total_steps
            alpha = args.domain_lambda * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)
            global_step += 1

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_domain = batch_domain.to(device)

            optimizer.zero_grad()
            label_logits, domain_logits = model(batch_x, alpha=alpha)
            label_loss = label_criterion(label_logits, batch_y)
            domain_loss = domain_criterion(domain_logits, batch_domain)
            loss = label_loss + args.domain_loss_weight * domain_loss
            loss.backward()
            if args.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()

            running_label_loss += label_loss.item() * batch_x.size(0)
            running_domain_loss += domain_loss.item() * batch_x.size(0)
            running_domain_correct += (domain_logits.argmax(dim=1) == batch_domain).sum().item()
            running_seen += batch_x.size(0)

        label_loss_epoch = running_label_loss / running_seen
        domain_loss_epoch = running_domain_loss / running_seen
        domain_acc_epoch = running_domain_correct / running_seen

        u_val_true, u_val_scores = predict_scores(model, x_u_val, y_u_val, args.batch_size, device)
        a_val_true, a_val_scores = predict_scores(model, x_a_val, y_a_val, args.batch_size, device)
        combined_val_true = np.concatenate([u_val_true, a_val_true])
        combined_val_scores = np.concatenate([u_val_scores, a_val_scores])
        threshold, _, constrained = mixed_trainer.best_threshold_balanced(
            combined_val_true, combined_val_scores, args.min_specificity
        )
        ucddb_threshold, _, _ = mixed_trainer.best_threshold_balanced(
            u_val_true, u_val_scores, args.min_specificity
        )

        u_metrics = mixed_trainer.evaluate_from_scores(u_val_true, u_val_scores, threshold=threshold)
        a_metrics = mixed_trainer.evaluate_from_scores(a_val_true, a_val_scores, threshold=threshold)
        selection_score = (
            args.ucddb_val_weight * u_metrics["balanced_accuracy"]
            + (1.0 - args.ucddb_val_weight) * a_metrics["balanced_accuracy"]
        )
        scheduler.step(selection_score)

        print(
            f"Epoch {epoch:02d} | LabelLoss={label_loss_epoch:.4f} DomainLoss={domain_loss_epoch:.4f} "
            f"DomainAcc={domain_acc_epoch:.4f} | thr={threshold:.3f}{'' if constrained else '*'} | "
            f"UCDDB Val BAcc={u_metrics['balanced_accuracy']:.4f} F1={u_metrics['f1']:.4f} "
            f"Rec={u_metrics['recall']:.4f} Spec={u_metrics['specificity']:.4f} | "
            f"Apnea Val BAcc={a_metrics['balanced_accuracy']:.4f} F1={a_metrics['f1']:.4f} "
            f"Rec={a_metrics['recall']:.4f} Spec={a_metrics['specificity']:.4f} | "
            f"Score={selection_score:.4f}"
        )

        if selection_score > best_score:
            best_score = selection_score
            best_threshold = threshold
            best_ucddb_threshold = ucddb_threshold
            best_epoch = epoch
            patience = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "settings": vars(args),
                    "best_threshold_combined_val": best_threshold,
                    "best_threshold_ucddb_val": best_ucddb_threshold,
                },
                model_path,
            )
        else:
            patience += 1
            if patience >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    def test_block(x, y):
        labels, scores = predict_scores(model, x, y, args.batch_size, device)
        return {
            "threshold_0_5": mixed_trainer.evaluate_from_scores(labels, scores, 0.5),
            "threshold_combined_val": mixed_trainer.evaluate_from_scores(labels, scores, best_threshold),
            "threshold_ucddb_val": mixed_trainer.evaluate_from_scores(labels, scores, best_ucddb_threshold),
        }

    results = {
        "settings": {
            **vars(args),
            "best_epoch": best_epoch,
            "normalization": "ECG per-window z-score after bandpass; repeated after augmentation",
        },
        "records": records,
        "record_stats": record_stats,
        "sample_summaries": {
            "ucddb_train": mixed_trainer.dataset_summary("ucddb_train", y_u_train),
            "ucddb_val": mixed_trainer.dataset_summary("ucddb_val", y_u_val),
            "ucddb_test": mixed_trainer.dataset_summary("ucddb_test", y_u_test),
            "apnea_train": mixed_trainer.dataset_summary("apnea_train", y_a_train),
            "apnea_val": mixed_trainer.dataset_summary("apnea_val", y_a_val),
            "apnea_test": mixed_trainer.dataset_summary("apnea_test", y_a_test),
            "combined_train": mixed_trainer.dataset_summary("combined_train", y_train),
        },
        "best_selection_score": best_score,
        "best_threshold_combined_val": best_threshold,
        "best_threshold_ucddb_val": best_ucddb_threshold,
        "ucddb_test": test_block(x_u_test, y_u_test),
        "apnea_test": test_block(x_a_test, y_a_test),
        "model_path": str(model_path),
    }

    result_path = output_dir / args.result_name
    result_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved model: {model_path}")
    print(f"Saved results: {result_path}")
    print("\nFinal holdout test metrics")
    for domain in ("ucddb_test", "apnea_test"):
        print(f"{domain}:")
        for threshold_name, metrics in results[domain].items():
            print(
                f"  {threshold_name}: Acc={metrics['accuracy']:.4f} "
                f"BAcc={metrics['balanced_accuracy']:.4f} F1={metrics['f1']:.4f} "
                f"Rec={metrics['recall']:.4f} Spec={metrics['specificity']:.4f} "
                f"AUC={metrics['roc_auc']:.4f}"
            )


def main():
    parser = argparse.ArgumentParser(description="Domain-adversarial mixed ECG training for wearable apnea detection.")
    parser.add_argument("--ucddb-dir", default="ucddb")
    parser.add_argument("--apnea-dir", default="apnea-ecg")
    parser.add_argument("--ucddb-cache-dir", default="aligned_data/ucddb")
    parser.add_argument("--apnea-cache-dir", default="aligned_data/apnea_ecg")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--model-name", default="dann_context5_weakaug.pth")
    parser.add_argument("--result-name", default="dann_context5_weakaug_results.json")
    parser.add_argument("--channels", nargs="+", type=int, default=[0])
    parser.add_argument("--context-minutes", type=int, default=5)
    parser.add_argument("--min-overlap-sec", type=float, default=1.0)
    parser.add_argument("--apnea-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--ucddb-sample-weight", type=float, default=2.0)
    parser.add_argument("--apnea-sample-weight", type=float, default=1.0)
    parser.add_argument("--ucddb-val-weight", type=float, default=0.7)
    parser.add_argument("--min-specificity", type=float, default=0.65)
    parser.add_argument("--domain-lambda", type=float, default=0.5)
    parser.add_argument("--domain-loss-weight", type=float, default=0.2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--flip-prob", type=float, default=0.10)
    parser.add_argument("--max-shift-sec", type=float, default=1.0)
    parser.add_argument("--scale-min", type=float, default=0.90)
    parser.add_argument("--scale-max", type=float, default=1.10)
    parser.add_argument("--noise-std-max", type=float, default=0.02)
    parser.add_argument("--baseline-amp-max", type=float, default=0.03)
    parser.add_argument("--mask-prob", type=float, default=0.10)
    parser.add_argument("--mask-max-sec", type=float, default=0.20)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
