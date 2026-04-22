"""Thin wrapper over deepcore.methods — builds the selector, calls .select()."""
import deepcore.methods as methods


# Map common/canonical names to the actual class names in deepcore.methods.
NAME_ALIASES = {
    "KCenterGreedy": "kCenterGreedy",
    "kcentergreedy": "kCenterGreedy",
    "Grand": "GraNd",
    "GRAND": "GraNd",
    "CRAIG": "Craig",
    "CAL": "Cal",
}


# Per-method kwargs that DeepCore expects via **kwargs.
# Conservative defaults — override via config's `selection_kwargs`.
METHOD_DEFAULTS = {
    "Uniform": dict(balance=True),
    "Herding": dict(balance=True),
    "kCenterGreedy": dict(balance=True),
    "Forgetting": dict(),
    "GraNd": dict(),
    "Cal": dict(balance=True),
    "Craig": dict(balance=True, greedy="LazyGreedy"),
    "GradMatch": dict(balance=True),
    "Glister": dict(balance=True, greedy="LazyGreedy"),
    "ContextualDiversity": dict(balance=True),
    "DeepFool": dict(balance=True),
    "Submodular": dict(balance=True, function="GraphCut", greedy="LazyGreedy"),
    "Uncertainty": dict(balance=True, selection_method="Entropy"),
    "Full": dict(),
    "CCS": dict(balance=False),
    "ClasswiseCCS": dict(),
}


def select_coreset(selection_name, dst_train, args, fraction, seed,
                   selection_kwargs=None):
    """
    selection_name: one of the classes in deepcore.methods
    Returns the subset dict {'indices': ndarray, optional 'weights'/'scores'}
    """
    resolved = NAME_ALIASES.get(selection_name, selection_name)
    kwargs = dict(METHOD_DEFAULTS.get(resolved, {}))
    if selection_kwargs:
        kwargs.update(selection_kwargs)

    # The 'epochs' key is consumed by EarlyTrain-based methods via **kwargs.
    kwargs.setdefault("epochs", args.selection_epochs)

    if resolved not in methods.__dict__:
        available = [k for k in methods.__dict__ if k[0].isupper() or k[0] == 'k']
        raise KeyError(
            f"Coreset method '{selection_name}' (resolved to '{resolved}') not found. "
            f"Available: {sorted(available)}"
        )
    cls = methods.__dict__[resolved]
    method = cls(dst_train, args, fraction, seed, **kwargs)
    return method.select()
