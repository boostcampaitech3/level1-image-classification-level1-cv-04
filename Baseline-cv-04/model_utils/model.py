import timm
import torch
import torch.nn as nn
from torchvision import models

from model_utils.custom_module import TripleHeadClassifier


def load_model(args):

    model_type = args['MODEL']
    num_classes = args['NUM_CLASSES']
    
    if model_type not in dir(timm.models):
        raise Exception(f'No model named {model_type}')
    
    model = timm.create_model(model_type, pretrained=True, num_classes=num_classes)

    if args['MULTI']:
        model = change_last_child_module_to_multihead(model, args)

    return model


def change_last_child_module_to_multihead(model, args):
    last_child_name, last_child_module = list(model.named_children())[-1]
    
    for m in last_child_module.modules():
        if hasattr(m, "in_features"):
            num_inputs = m.in_features
            break
        elif hasattr(m, "in_channels"):
            num_inputs = m.in_channels
            break
        
    exec(f"model.{last_child_name} = TripleHeadClassifier({num_inputs}, {args['MULTI']})")

    return model


### torchvision model 사용 예시 ###
# if model_type == 'efficientnet-b3':

#     model = models.resnet18(pretrained=True)
#     model.fc = nn.Linear(model.fc.in_features, num_classes)