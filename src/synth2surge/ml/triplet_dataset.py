"""Triplet dataset for training the audio encoder.

Mines triplets from optimization trial data: (anchor=target, positive=low-loss
candidate, negative=higher-loss candidate). The encoder learns that low-loss
candidates should embed closer to the target than high-loss ones.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    from torch.utils.data import Dataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class TripletDataset(Dataset):
        """Dataset of (anchor, positive, negative) audio feature triplets.

        Mined from optimization trial data stored in the experience store.
        Each triplet is: target audio features, a close-match trial, a far-match trial.

        For now, uses the 512-dim feature vectors rather than raw spectrograms
        for efficiency. Can be upgraded to raw mel-spectrograms later.
        """

        def __init__(
            self,
            store_path: Path,
            *,
            min_loss_gap: float = 0.5,
            max_triplets_per_run: int = 50,
        ) -> None:
            """
            Args:
                store_path: Path to experience store database.
                min_loss_gap: Minimum loss difference between positive and negative.
                max_triplets_per_run: Max triplets to mine from each optimization run.
            """
            from synth2surge.ml.experience_store import ExperienceStore

            store = ExperienceStore(store_path)
            self.triplets: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

            # Get all runs that have trials logged
            conn = store._conn
            run_ids = [
                row[0]
                for row in conn.execute(
                    "SELECT DISTINCT run_id FROM trials"
                ).fetchall()
            ]

            for run_id in run_ids:
                run = store.get_run(run_id)
                if run is None:
                    continue

                anchor = run.target_features

                # Get trials sorted by loss
                trial_rows = conn.execute(
                    "SELECT params, loss FROM trials WHERE run_id = ? ORDER BY loss ASC",
                    (run_id,),
                ).fetchall()

                if len(trial_rows) < 2:
                    continue

                from synth2surge.ml.experience_store import _blob_to_array

                trials = [(_blob_to_array(p), loss) for p, loss in trial_rows]

                # Mine triplets: pick positive from low-loss, negative from higher-loss
                n_mined = 0
                n_trials = len(trials)
                for i in range(min(n_trials // 4, max_triplets_per_run)):
                    pos_idx = i
                    # Find a negative with sufficient loss gap
                    pos_loss = trials[pos_idx][1]
                    for neg_idx in range(n_trials - 1, pos_idx, -1):
                        neg_loss = trials[neg_idx][1]
                        if neg_loss - pos_loss >= min_loss_gap:
                            self.triplets.append((
                                anchor,
                                trials[pos_idx][0],
                                trials[neg_idx][0],
                            ))
                            n_mined += 1
                            break

                    if n_mined >= max_triplets_per_run:
                        break

            store.close()
            logger.info(
                f"Mined {len(self.triplets)} triplets from {len(run_ids)} runs"
            )

        def __len__(self) -> int:
            return len(self.triplets)

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            anchor, positive, negative = self.triplets[idx]
            return (
                torch.tensor(anchor, dtype=torch.float32),
                torch.tensor(positive, dtype=torch.float32),
                torch.tensor(negative, dtype=torch.float32),
            )
