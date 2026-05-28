import json
import tempfile
import zipfile
from pathlib import Path

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="CancerPrediction")
class ResNet50Preprocessing(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.keras.applications.resnet50.preprocess_input(inputs * 255.0)


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


def _repair_layer_config(layer):
    layer_config = layer.get("config", {})

    if layer.get("class_name") != "Lambda":
        return False

    if layer.get("name") == "resnet50_preprocessing" or layer_config.get("name") == "resnet50_preprocessing":
        layer["module"] = "src.utils.model_io"
        layer["class_name"] = "ResNet50Preprocessing"
        layer["registered_name"] = "CancerPrediction>ResNet50Preprocessing"
        layer["config"] = {
            "name": layer_config.get("name", "resnet50_preprocessing"),
            "trainable": layer_config.get("trainable", True),
            "dtype": layer_config.get("dtype", "float32"),
        }
        return True

    if layer_config.get("output_shape") is None:
        layer_config["output_shape"] = [None, None, 3]
        return True

    return False


def _write_portable_repaired_copy(model_path):
    model_path = Path(model_path)
    repaired_path = Path(tempfile.gettempdir()) / f"{model_path.stem}_portable_repaired.keras"

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
                    changed = _repair_layer_config(layer) or changed

                if changed:
                    data = json.dumps(config).encode("utf-8")

            target.writestr(item, data)

    return repaired_path


def load_keras_model(model_path):
    custom_objects = {
        "swish": tf.keras.activations.swish,
        "tf": tf,
        "ResNet50Preprocessing": ResNet50Preprocessing,
        "CancerPrediction>ResNet50Preprocessing": ResNet50Preprocessing,
    }

    try:
        model = tf.keras.models.load_model(model_path, safe_mode=False, custom_objects=custom_objects)
        return _inject_tensorflow_into_lambda_layers(model)
    except (NotImplementedError, TypeError, ValueError) as exc:
        error_text = str(exc)
        if (
            "Lambda" not in error_text
            and "output_shape" not in error_text
            and "marshal" not in error_text
            and "bad marshal data" not in error_text
        ):
            raise

        repaired_path = _write_portable_repaired_copy(model_path)
        model = tf.keras.models.load_model(repaired_path, safe_mode=False, custom_objects=custom_objects)
        return _inject_tensorflow_into_lambda_layers(model)
