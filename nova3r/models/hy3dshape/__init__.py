# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

"""Hunyuan3D components with inference-only dependencies loaded lazily."""

from importlib import import_module


_LAZY_EXPORTS = {
    "Hunyuan3DDiTPipeline": (".pipelines", "Hunyuan3DDiTPipeline"),
    "Hunyuan3DDiTFlowMatchingPipeline": (
        ".pipelines",
        "Hunyuan3DDiTFlowMatchingPipeline",
    ),
    "FaceReducer": (".postprocessors", "FaceReducer"),
    "FloaterRemover": (".postprocessors", "FloaterRemover"),
    "DegenerateFaceRemover": (".postprocessors", "DegenerateFaceRemover"),
    "MeshSimplifier": (".postprocessors", "MeshSimplifier"),
    "ImageProcessorV2": (".preprocessors", "ImageProcessorV2"),
    "IMAGE_PROCESSORS": (".preprocessors", "IMAGE_PROCESSORS"),
    "DEFAULT_IMAGEPROCESSOR": (".preprocessors", "DEFAULT_IMAGEPROCESSOR"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name):
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
