import torch
import torch.nn as nn
import timm


class SE(nn.Module):
    """Squeeze and Excitation Block

    출처: https://sseunghyuns.github.io/classification/2021/09/05/blindness-detection/
    """
    def __init__(self, num_channels, reduction_ratio=2):
        super().__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        batch_size, num_channels, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor


class TripleHeadClassifier(nn.Module):
    def __init__(self, num_features, nums_multihead, use_globalpool=False):
        assert len(nums_multihead) == 3, f"MultiHead의 길이가 {len(nums_multihead)}입니다."
        super(TripleHeadClassifier, self).__init__()
        n_m, n_g, n_a = nums_multihead

        self.use_globalpool = use_globalpool
        if self.use_globalpool:
            self.globalavgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        # self.mask_classifier = timm.models.layers.ClassifierHead(num_features, n_m)
        self.mask_classifier = nn.Linear(num_features, n_m)
        # self.gender_classifier = timm.models.layers.ClassifierHead(num_features, n_g)
        self.gender_classifier = nn.Linear(num_features, n_g)
        # self.age_classifier = timm.models.layers.ClassifierHead(num_features, n_a)
        self.age_classifier = nn.Linear(num_features, n_a)

    def forward(self, x):
        if self.use_globalpool:
            x = self.globalavgpool(x)
        x = self.flatten(x)
        mask = self.mask_classifier(x)
        gender = self.gender_classifier(x)
        age = self.age_classifier(x)

        return mask, gender, age
    