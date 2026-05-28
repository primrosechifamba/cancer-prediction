import json
import tempfile
import zipfile
from pathlib import Path

import tensorflow as tf


def _inject_tensorflow_into_lambda_layers(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Lambda) and hasattr(layer, "function"):
            try:
                layer.function.__globals__.setdefault("tf", tf)
            except AttributeError:
                pass

        if isinstance(layer, tf.keras.Model):
            _inject_tensorflow_into_lambda_layers(layer)

    return model


def _write_lambda_shape_repaired_copy(model_path):
    model_path = Path(model_path)
    repaired_path = Path(tempfile.gettempdir()) / f"{model_path.stem}_lambda_repaired.keras"

    with zipfile.ZipFile(model_path, "r") as source, zipfile.ZipFile(
        repaired_path,
        "w",
        zipfile.ZIP_DEFLATED,
    ) as target:
        for item in source.infolist():
            data = source.read(item.filename)

            if item.filename == "config.json":
                config = json.loads(data)
                changed = False

                for layer in config.get("config", {}).get("layers", []):
                    layer_config = layer.get("config", {})
                    if layer.get("class_name") == "Lambda" and layer_config.get("output_shape") is None:
                        layer_config["output_shape"] = [None, None, 3]
                        changed = True

                if changed:
                    data = json.dumps(config).encode("utf-8")

            target.writestr(item, data)

    return repaired_path


def load_keras_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path, safe_mode=False)
        return _inject_tensorflow_into_lambda_layers(model)
    except NotImplementedError as exc:
        if "Lambda" not in str(exc) and "output_shape" not in str(exc):
            raise

        repaired_path = _write_lambda_shape_repaired_copy(model_path)
        model = tf.keras.models.load_model(repaired_path, safe_mode=False)
        return _inject_tensorflow_into_lambda_layers(model)
    except TypeError as exc:
        # Keras may fail to resolve custom objects (e.g. 'swish' activation or
        # references inside Lambda layers). Retry loading with a conservative
        # set of common custom objects and inject `tf` into lambda globals.
        try:
            custom_objects = {
                "swish": tf.keras.activations.swish,
                "tf": tf,
            }
            model = tf.keras.models.load_model(model_path, safe_mode=False, custom_objects=custom_objects)
            return _inject_tensorflow_into_lambda_layers(model)
        except Exception:
            # Re-raise the original TypeError for visibility if fallback fails
            raise
