"""Bradley--Terry-inspired aggregation for VLM trajectory comparisons."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class PairwisePreference:
    left_id: str
    right_id: str
    winner: str
    confidence: str
    evidence: str
    raw_response: str = ""


def parse_pairwise_response(
    response: str,
    left_id: str,
    right_id: str,
) -> PairwisePreference:
    """Parse the deliberately small VLM comparison schema robustly."""
    preferred_match = re.search(
        r"(?:preferred|preference|winner)\s*:\s*(policy\s*)?(a|b|tie|equal)",
        response,
        flags=re.IGNORECASE,
    )
    token = preferred_match.group(2).lower() if preferred_match else "tie"
    if token == "a":
        winner = left_id
    elif token == "b":
        winner = right_id
    else:
        winner = "tie"

    confidence_match = re.search(
        r"confidence\s*:\s*(low|medium|high)", response, flags=re.IGNORECASE
    )
    confidence = confidence_match.group(1).lower() if confidence_match else "low"

    evidence_match = re.search(
        r"evidence\s*:\s*(.+)", response, flags=re.IGNORECASE | re.DOTALL
    )
    evidence = evidence_match.group(1).strip() if evidence_match else response.strip()
    return PairwisePreference(
        left_id=left_id,
        right_id=right_id,
        winner=winner,
        confidence=confidence,
        evidence=evidence,
        raw_response=response,
    )


class BradleyTerryAggregator:
    """Fit latent utilities internally and expose only a score-free ranking."""

    _CONFIDENCE_WEIGHT = {"low": 0.5, "medium": 1.0, "high": 2.0}

    def __init__(
        self,
        learning_rate: float = 0.08,
        iterations: int = 300,
        l2_regularization: float = 1e-3,
    ):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.l2_regularization = l2_regularization

    def rank(
        self,
        item_ids: Sequence[str],
        comparisons: Iterable[PairwisePreference],
    ) -> list[str]:
        """Return IDs ordered by fitted preference, without returning BT scores."""
        ids = list(dict.fromkeys(item_ids))
        if not ids:
            return []
        index = {item_id: idx for idx, item_id in enumerate(ids)}
        valid = [
            comparison
            for comparison in comparisons
            if comparison.left_id in index and comparison.right_id in index
        ]
        if not valid:
            return ids

        utilities = np.zeros(len(ids), dtype=float)
        for _ in range(self.iterations):
            gradient = -self.l2_regularization * utilities
            for comparison in valid:
                left = index[comparison.left_id]
                right = index[comparison.right_id]
                if comparison.winner == comparison.left_id:
                    target = 1.0
                elif comparison.winner == comparison.right_id:
                    target = 0.0
                else:
                    target = 0.5
                difference = float(np.clip(utilities[left] - utilities[right], -30.0, 30.0))
                probability = 1.0 / (1.0 + np.exp(-difference))
                weight = self._CONFIDENCE_WEIGHT.get(comparison.confidence, 0.5)
                residual = weight * (target - probability)
                gradient[left] += residual
                gradient[right] -= residual
            utilities += self.learning_rate * gradient / max(1, len(valid))
            utilities -= np.mean(utilities)

        order = np.argsort(utilities)[::-1]
        return [ids[idx] for idx in order]
