import numpy as np
from .ccs import CCS


class ClasswiseCCS(CCS):
    """
    Class-wise Coverage-Centric Coreset Selection.

    Distributes the total coreset budget across classes to make the selected
    class distribution as uniform as possible (each class gets at most
    floor(budget / num_classes) samples to start, with remaining budget
    assigned to classes that still have capacity, prioritised by class size).
    Within each class the standard CCS strategy is applied.
    """

    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, epochs=200,
                 specific_model=None, **kwargs):
        kwargs.pop('balance', None)
        super().__init__(dst_train, args, fraction, random_seed, epochs=epochs,
                         specific_model=specific_model, balance=False, **kwargs)

    def _allocate_budget(self, targets):
        """Return per-class budgets that are as equal as possible given class sizes."""
        class_counts = np.array([int((targets == c).sum()) for c in range(self.num_classes)])
        budget = self.coreset_size

        per_class = np.minimum(class_counts, budget // self.num_classes)
        remainder = budget - int(per_class.sum())

        capacity = class_counts - per_class
        while remainder > 0 and capacity.sum() > 0:
            eligible = np.where(capacity > 0)[0]
            give = min(remainder, len(eligible))
            top = np.argsort(-capacity[eligible])[:give]
            per_class[eligible[top]] += 1
            capacity[eligible[top]] -= 1
            remainder -= give

        return per_class

    def select(self, **kwargs):
        self.run()
        targets = np.array(self.dst_train.targets)
        budgets = self._allocate_budget(targets)

        selected = []
        for c in range(self.num_classes):
            if budgets[c] == 0:
                continue
            class_idx = np.where(targets == c)[0]
            selected.append(self._ccs_select(class_idx, int(budgets[c])))

        return {"indices": np.concatenate(selected)}
