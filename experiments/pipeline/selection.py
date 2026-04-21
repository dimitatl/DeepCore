"""Thin wrapper over deepcore.methods — builds the selector, calls .select()."""
import deepcore.methods as methods


# Per-method kwargs that DeepCore expects via **kwargs.
# Conservative defaults — override via config's `selection_kwargs`.
METHOD_DEFAULTS = {
    "Uniform": dict(balance=True),
    "Herding": dict(balance=True),
    "KCenterGreedy": dict(balance=True),
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
}


def select_coreset(selection_name, dst_train, args, fraction, seed,
                   selection_kwargs=None):
    """
    selection_name: one of the classes in deepcore.methods
    Returns the subset dict {'indices': ndarray, optional 'weights'/'scores'}
    """
    kwargs = dict(METHOD_DEFAULTS.get(selection_name, {}))
    if selection_kwargs:
        kwargs.update(selection_kwargs)

    # The 'epochs' key is consumed by EarlyTrain-based methods via **kwargs.
    kwargs.setdefault("epochs", args.selection_epochs)

    cls = methods.__dict__[selection_name]
    method = cls(dst_train, args, fraction, seed, **kwargs)
    return method.select()
