from .nodes import NormalMapLightEstimator

NODE_CLASS_MAPPINGS = {
    "NormalMapLightEstimator": NormalMapLightEstimator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NormalMapLightEstimator": "Normal Map Light Estimator"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
