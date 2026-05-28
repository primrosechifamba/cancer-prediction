from src.models.resnet50 import build_resnet50
from src.models.mobilenetv2 import build_mobilenetv2
from src.models.efficientnetb0 import build_efficientnetb0

# =========================================================
# MODEL FACTORY
# =========================================================

def get_model(model_name, input_shape, num_classes):

    if model_name.lower() == "resnet50":

        return build_resnet50(
            input_shape,
            num_classes
        )

    elif model_name.lower() == "mobilenetv2":

        return build_mobilenetv2(
            input_shape,
            num_classes
        )

    elif model_name.lower() == "efficientnetb0":

        return build_efficientnetb0(
            input_shape,
            num_classes
        )

    else:

        raise ValueError(
            f"Unknown Model: {model_name}"
        )