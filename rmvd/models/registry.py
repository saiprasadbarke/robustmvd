"""Model Registry.

Based on the model registry from the timm package ( https://github.com/huggingface/pytorch-image-models ).
"""


_model_entrypoints = {}  # store registered models
_trainable_model_entrypoints = {}  # store registered trainable models


def register_model(arg=None, trainable=True):
    def _register_model(model_entrypoint):
        """Register a model."""
        model_name = model_entrypoint.__name__
        _model_entrypoints[model_name] = model_entrypoint
        # If the model is trainable, add it to the trainable models as well
        if trainable:
            _trainable_model_entrypoints[model_name] = model_entrypoint
        return model_entrypoint

    if callable(arg):
        return _register_model(arg)
    else:
        return _register_model


def list_models(trainable_only=False):
    """List all available models."""
    models = (
        _model_entrypoints.keys()
        if not trainable_only
        else _trainable_model_entrypoints.keys()
    )
    models = list(sorted(models))
    return models


def has_model(name, trainable_only=False):
    """Check if model is registered."""
    has_model = (
        name in _model_entrypoints
        if not trainable_only
        else name in _trainable_model_entrypoints
    )
    return has_model


def get_model(name):
    """Get model entrypoint by name."""
    assert has_model(
        name
    ), f'The requested model "{name}" does not exist. Available models are: {" ".join(list_models())}'
    return _model_entrypoints[name]
