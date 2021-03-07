import cloudpickle
import pytorch_lightning
import torch
import torchvision
import albumentations
import efficientnet_pytorch

CONDA_ENV = {
        'name': 'mlflow-env',
        'channels': ['defaults', 'conda-forge', 'pytorch'],
        'dependencies': [
            'python=3.7.7',
            f'cloudpickle=={cloudpickle.__version__}',
            f'pytorch=={torch.__version__}',
            'mlflow',
            f'torchvision=={torchvision.__version__}',
            f'pytorch-lightning=={pytorch_lightning.__version__}',
            f'albumentations=={albumentations.__version__}',
            {
                'pip': [
                    f'efficientnet_pytorch=={efficientnet_pytorch.__version__}',
                ]
            }
        ]
    }