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
    children = list(model.named_children())[::-1]
    last_child_name, last_child_module = children[0]
    
    # 마지막 child_module에 인접한 Conv / Pool / Linear 레이어 검색
    found = False
    for _, child_module in children[1:]:
        for m in list(child_module.modules())[::-1]:
            module_type = str(type(m))
            if "Conv2d" in module_type or "Pool2d" in module_type or "Linear" in module_type:
                found = True; break
        if found: break
    use_globalpool = "Conv2d" in module_type # 인접한 child_module이 Conv이면 global average pool 수행

    for m in last_child_module.modules():
        if hasattr(m, "in_features"):
            num_inputs = m.in_features
            break
        elif hasattr(m, "in_channels"):
            num_inputs = m.in_channels
            break
        
    exec(f"model.{last_child_name} = TripleHeadClassifier({num_inputs}, {args['MULTI']}, use_globalpool={use_globalpool})")

    return model


### torchvision model 사용 예시 ###
# if model_type == 'efficientnet-b3':

#     model = models.resnet18(pretrained=True)
#     model.fc = nn.Linear(model.fc.in_features, num_classes)