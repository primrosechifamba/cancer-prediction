import json
import tempfile
import zipfile
from pathlib import Path

import tensorflow as tf


def _deserialize_dtype(dtype):
    if isinstance(dtype, dict):
        try:
            return tf.keras.utils.deserialize_keras_object(dtype)
        except (TypeError, ValueError):
            return dtype.get("config", {}).get("name", "float32")
    return dtype


@tf.keras.utils.register_keras_serializable(package="CancerPrediction")
class ResNet50Preprocessing(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        kwargs["dtype"] = _deserialize_dtype(kwargs.get("dtype"))
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        return tf.keras.applications.resnet50.preprocess_input(inputs * 255.0)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return super().get_config()

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config["dtype"] = _deserialize_dtype(config.get("dtype"))
        return cls(**config)


@tf.keras.utils.register_keras_serializable(package="CancerPrediction")
class MobileNetV2Preprocessing(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        kwargs["dtype"] = _deserialize_dtype(kwargs.get("dtype"))
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        return tf.keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return super().get_config()

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config["dtype"] = _deserialize_dtype(config.get("dtype"))
        return cls(**config)


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
    changed = False

    if "quantization_config" in layer_config:
        layer_config.pop("quantization_config", None)
        changed = True

    if layer.get("class_name") == "InputLayer" and "optional" in layer_config:
        layer_config.pop("optional", None)
        changed = True

    if layer.get("class_name") == "Functional":
        for key in ("input_layers", "output_layers"):
            endpoint = layer_config.get(key)
            if _is_flat_keras_endpoint(endpoint):
                layer_config[key] = [endpoint]
                changed = True

    if layer.get("class_name") == "BatchNormalization":
        for key in ("renorm", "renorm_clipping", "renorm_momentum"):
            if key in layer_config:
                layer_config.pop(key, None)
                changed = True

    if layer.get("class_name") != "Lambda":
        return changed

    dtype = _deserialize_dtype(layer_config.get("dtype", "float32"))
    if hasattr(dtype, "name"):
        dtype = dtype.name

    if layer.get("name") == "resnet50_preprocessing" or layer_config.get("name") == "resnet50_preprocessing":
        layer["module"] = None
        layer["class_name"] = "ResNet50Preprocessing"
        layer["registered_name"] = "CancerPrediction>ResNet50Preprocessing"
        layer["config"] = {
            "name": layer_config.get("name", "resnet50_preprocessing"),
            "trainable": layer_config.get("trainable", True),
            "dtype": dtype,
        }
        return True

    if layer.get("name") == "mobilenetv2_preprocessing" or layer_config.get("name") == "mobilenetv2_preprocessing":
        layer["module"] = None
        layer["class_name"] = "MobileNetV2Preprocessing"
        layer["registered_name"] = "CancerPrediction>MobileNetV2Preprocessing"
        layer["config"] = {
            "name": layer_config.get("name", "mobilenetv2_preprocessing"),
            "trainable": layer_config.get("trainable", True),
            "dtype": dtype,
        }
        return True

    if layer_config.get("output_shape") is None:
        layer_config["output_shape"] = [None, None, 3]
        return True

    return changed


def _is_flat_keras_endpoint(endpoint):
    return (
        isinstance(endpoint, list)
        and len(endpoint) == 3
        and isinstance(endpoint[0], str)
        and isinstance(endpoint[1], int)
        and isinstance(endpoint[2], int)
    )


def _repair_config_tree(value):
    changed = False

    if isinstance(value, dict):
        if "class_name" in value and "config" in value:
            changed = _repair_layer_config(value) or changed

        for child in value.values():
            changed = _repair_config_tree(child) or changed

    elif isinstance(value, list):
        for child in value:
            changed = _repair_config_tree(child) or changed

    return changed


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
                changed = _repair_config_tree(config)

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
        "src.utils.model_io.ResNet50Preprocessing": ResNet50Preprocessing,
        "MobileNetV2Preprocessing": MobileNetV2Preprocessing,
        "CancerPrediction>MobileNetV2Preprocessing": MobileNetV2Preprocessing,
        "src.utils.model_io.MobileNetV2Preprocessing": MobileNetV2Preprocessing,
    }

    load_path = model_path
    try:
        load_path = _write_portable_repaired_copy(model_path)
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile):
        load_path = model_path

    try:
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(
                load_path,
                safe_mode=False,
                custom_objects=custom_objects,
                compile=False,
            )
        return _inject_tensorflow_into_lambda_layers(model)
    except (NotImplementedError, TypeError, ValueError) as exc:
        error_text = str(exc)
        if (
            "Lambda" not in error_text
            and "output_shape" not in error_text
            and "marshal" not in error_text
            and "bad marshal data" not in error_text
            and "BatchNormalization" not in error_text
            and "renorm" not in error_text
            and "quantization_config" not in error_text
            and "InputLayer" not in error_text
            and "optional" not in error_text
        ):
            raise

        repaired_path = _write_portable_repaired_copy(model_path)
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(
                repaired_path,
                safe_mode=False,
                custom_objects=custom_objects,
                compile=False,
            )
        return _inject_tensorflow_into_lambda_layers(model)
