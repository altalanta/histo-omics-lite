from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

__all__ = ["RetrievalMetrics", "compute_retrieval_metrics"]


@dataclass(frozen=True)
class RetrievalMetrics:
    top1: float
    top5: float
    auroc: float

    def as_dict(self) -> dict[str, float]:
        return {"top1": self.top1, "top5": self.top5, "auroc": self.auroc}


def compute_retrieval_metrics(
    image_embeddings: torch.Tensor,
    omics_embeddings: torch.Tensor,
    labels: torch.Tensor,
) -> RetrievalMetrics:
    if image_embeddings.shape != omics_embeddings.shape:
        raise ValueError("Image and omics embeddings must share shape")
    if image_embeddings.shape[0] != labels.shape[0]:
        raise ValueError("Label vector must align with embeddings")

    with torch.no_grad():
        image_embeddings = torch.nn.functional.normalize(image_embeddings, dim=1)
        omics_embeddings = torch.nn.functional.normalize(omics_embeddings, dim=1)
        logits = image_embeddings @ omics_embeddings.T

    values, indices = torch.topk(logits, k=min(5, logits.size(1)), dim=1)
    matches = torch.arange(logits.size(0), device=logits.device)

    top1_hits = (indices[:, 0] == matches).float().mean().item()
    top5_hits = (indices == matches.unsqueeze(1)).any(dim=1).float().mean().item()

    pos_scores = logits.diag().cpu().numpy()
    neg_mask = ~torch.eye(logits.size(0), dtype=torch.bool, device=logits.device)
    neg_scores = logits[neg_mask].cpu().numpy()

    y_true = np.concatenate(
        [
            np.ones_like(pos_scores, dtype=np.int32),
            np.zeros_like(neg_scores, dtype=np.int32),
        ]
    )
    y_scores = np.concatenate([pos_scores, neg_scores])
    auroc = float(roc_auc_score(y_true, y_scores))

    return RetrievalMetrics(top1=float(top1_hits), top5=float(top5_hits), auroc=auroc)
