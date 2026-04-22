from .earlytrain import EarlyTrain
from ..nets.nets_utils import MyDataParallel
import torch
import numpy as np


class CCS(EarlyTrain):
    """
    Coverage-Centric Coreset Selection.
    https://github.com/haizhongzheng/Coverage-centric-coreset-selection

    Stratifies training samples by difficulty (per-sample error count over
    training epochs) into `coreset_size` equal-sized bins, then selects the
    sample closest to the centroid of each bin in embedding space.  This
    balances representation across the easy-to-hard difficulty spectrum while
    ensuring embedding-space coverage within each stratum.
    """

    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, epochs=200,
                 specific_model=None, balance=False, **kwargs):
        super().__init__(dst_train, args, fraction, random_seed, epochs=epochs,
                         specific_model=specific_model, **kwargs)
        self.balance = balance

    def before_run(self):
        self.emb_dim = self.model.get_last_layer().in_features
        self.error_count = torch.zeros(self.n_train, requires_grad=False).to(self.args.device)

    def after_loss(self, outputs, loss, targets, batch_inds, epoch):
        with torch.no_grad():
            preds = outputs.argmax(dim=1)
            wrong = (preds != targets).float()
            self.error_count[torch.tensor(batch_inds).to(self.args.device)] += wrong

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        if batch_idx % self.args.print_freq == 0:
            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f' % (
                epoch, self.epochs, batch_idx + 1, (self.n_train // batch_size) + 1, loss.item()))

    def finish_run(self):
        if isinstance(self.model, MyDataParallel):
            self.model = self.model.module
        self._all_embeddings = self._extract_embeddings()

    def _extract_embeddings(self, index=None):
        self.model.eval()
        self.model.no_grad = True
        with torch.no_grad():
            with self.model.embedding_recorder:
                parts = []
                data_loader = torch.utils.data.DataLoader(
                    self.dst_train if index is None else
                    torch.utils.data.Subset(self.dst_train, index),
                    batch_size=self.args.selection_batch,
                    num_workers=self.args.workers,
                )
                for inputs, _ in data_loader:
                    self.model(inputs.to(self.args.device))
                    parts.append(self.model.embedding_recorder.embedding.cpu().numpy())
        self.model.no_grad = False
        return np.concatenate(parts, axis=0)

    def _ccs_select(self, pool_indices, budget):
        """Select `budget` samples from `pool_indices` using difficulty-stratified centroid selection."""
        n = len(pool_indices)
        if budget >= n:
            return pool_indices

        embeddings = self._all_embeddings[pool_indices]
        difficulties = self.error_count[pool_indices].cpu().numpy()

        # Sort easy → hard
        order = np.argsort(difficulties, kind='stable')
        sorted_indices = pool_indices[order]
        sorted_embs = embeddings[order]

        # Partition into `budget` equal-sized bins and pick the centroid-nearest sample from each
        bin_edges = np.round(np.linspace(0, n, budget + 1)).astype(int)

        selected = []
        for j in range(budget):
            start, end = bin_edges[j], bin_edges[j + 1]
            if start == end:
                continue
            bin_embs = sorted_embs[start:end]
            centroid = bin_embs.mean(axis=0, keepdims=True)
            dists = np.sum((bin_embs - centroid) ** 2, axis=1)
            selected.append(sorted_indices[start + int(np.argmin(dists))])

        return np.array(selected, dtype=np.int64)

    def select(self, **kwargs):
        self.run()
        targets = np.array(self.dst_train.targets)

        if self.balance:
            selected = []
            for c in range(self.num_classes):
                class_idx = np.where(targets == c)[0]
                budget = round(self.fraction * len(class_idx))
                if budget > 0:
                    selected.append(self._ccs_select(class_idx, budget))
            return {"indices": np.concatenate(selected)}

        all_idx = np.arange(self.n_train)
        return {"indices": self._ccs_select(all_idx, self.coreset_size)}
