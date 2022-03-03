import timm
import torch
import torch.nn as nn
from torchvision import models

from model_utils.custom_module import TripleHeadClassifier


def load_model(args, num_classes):

    model_type = args['MODEL']
    
    if model_type not in dir(timm.models):
        raise Exception(f'No model named {model_type}')
    
    model = timm.create_model(model_type, pretrained=True, num_classes=num_classes)

    return model