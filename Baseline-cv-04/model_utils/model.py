import timm
import torch
import torch.nn as nn
from torchvision import models

def load_model(args):

    model_type = args['MODEL']
    num_classes = args['NUM_CLASSES']
    
    if model_type not in dir(timm.models):
        raise Exception(f'No model named {model_type}')
    
    model = timm.create_model(model_type, pretrained=True, num_classes=num_classes)
        

    return model


### torchvision model 사용 예시 ###
# if model_type == 'efficientnet-b3':

#     model = models.resnet18(pretrained=True)
#     model.fc = nn.Linear(model.fc.in_features, num_classes)
