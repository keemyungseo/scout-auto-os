"""Classification metrics for lifecycle labels."""

from __future__ import annotations

from collections import defaultdict


def confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
) -> list[dict]:
    idx = {lab: i for i, lab in enumerate(labels)}
    mat = [[0 for _ in labels] for _ in labels]
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            mat[idx[t]][idx[p]] += 1
    rows: list[dict] = []
    for i, lab in enumerate(labels):
        row = {"true_label": lab}
        for j, plab in enumerate(labels):
            row[f"pred_{plab}"] = mat[i][j]
        row["true_total"] = sum(mat[i])
        rows.append(row)
    return rows


def per_class_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
) -> list[dict]:
    rows: list[dict] = []
    for lab in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p == lab)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lab and p == lab)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p != lab)
        support = sum(1 for t in y_true if t == lab)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append(
            {
                "label": lab,
                "support": support,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            },
        )
    return rows


def aggregate_metrics(per_class: list[dict]) -> dict:
    supports = [r["support"] for r in per_class if r["support"] > 0]
    total = sum(supports) or 1
    macro_p = sum(r["precision"] for r in per_class if r["support"] > 0) / max(len(supports), 1)
    macro_r = sum(r["recall"] for r in per_class if r["support"] > 0) / max(len(supports), 1)
    macro_f1 = sum(r["f1"] for r in per_class if r["support"] > 0) / max(len(supports), 1)
    w_p = sum(r["precision"] * r["support"] for r in per_class) / total
    w_r = sum(r["recall"] * r["support"] for r in per_class) / total
    w_f1 = sum(r["f1"] * r["support"] for r in per_class) / total
    return {
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_precision": round(w_p, 4),
        "weighted_recall": round(w_r, 4),
        "weighted_f1": round(w_f1, 4),
    }


def binary_discrimination(
    y_true: list[str],
    y_pred: list[str],
    positive: str,
    negative: str,
) -> dict:
    """Measure separation between two lifecycle types (e.g. Fake vs Continuous)."""
    tp = fp = tn = fn = 0
    for t, p in zip(y_true, y_pred):
        t_pos = t == positive
        p_pos = p == positive
        if t_pos and p_pos:
            tp += 1
        elif not t_pos and p_pos:
            fp += 1
        elif t_pos and not p_pos:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    acc = (tp + tn) / max(len(y_true), 1)
    return {
        "positive": positive,
        "negative": negative,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(acc, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def top_k_accuracy(proba_rows: list[dict[str, float]], y_true: list[str], k: int = 2) -> float:
    hits = 0
    for probs, lab in zip(proba_rows, y_true):
        ranked = sorted(probs.items(), key=lambda x: -x[1])
        top = {x[0] for x in ranked[:k]}
        if lab in top:
            hits += 1
    return round(hits / max(len(y_true), 1), 4)
