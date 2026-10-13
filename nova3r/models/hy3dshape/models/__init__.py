# Open Source Model Licensed under the Apache License Version 2.0
# and Other Licenses of the Third-Party Components therein:
# The below Model in this distribution may have been modified by THL A29 Limited
# ("Tencent Modifications"). All Tencent Modifications are Copyright (C) 2024 THL A29 Limited.

# Copyright (C) 2024 THL A29 Limited, a Tencent company.  All rights reserved.
# The below software and/or models in this distribution may have been
# modified by THL A29 Limited ("Tencent Modifications").
# All Tencent Modifications are Copyright (C) THL A29 Limited.

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


"""Model exports without importing optional image and mesh dependencies."""

from importlib import import_module


_LAZY_EXPORTS = {
    "ShapeVAE": (
        "nova3r.heads.hunyuan_model.autoencoders",
        "ShapeVAE",
    ),
    "DualImageEncoder": (".conditioner", "DualImageEncoder"),
    "SingleImageEncoder": (".conditioner", "SingleImageEncoder"),
    "DinoImageEncoder": (".conditioner", "DinoImageEncoder"),
    "CLIPImageEncoder": (".conditioner", "CLIPImageEncoder"),
    "Hunyuan3DDiT": (".denoisers", "Hunyuan3DDiT"),
    "HunYuanDiTPlain": (".denoisers", "HunYuanDiTPlain"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name):
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    if module_name.startswith("."):
        value = getattr(import_module(module_name, __name__), attribute)
    else:
        value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
