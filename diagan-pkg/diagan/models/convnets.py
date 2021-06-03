"""ReBias
Copyright (c) 2020-present NAVER Corp.
MIT license
Implementation for simple statcked convolutional networks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
class SimpleConvNet(nn.Module):
    def __init__(self, num_labels = 10, num_channels = 3, kernel_size=7, **kwargs):
        super(SimpleConvNet, self).__init__()
        padding = kernel_size // 2
        layers = [
            nn.Conv2d(num_channels, 16, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        ]
        self.extracter = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_labels)
        self.dim_in = 128
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x = self.extracter(x)
        x = self.avgpool(x)
        feat = torch.flatten(x, 1)
        logits = self.fc(feat)
        feat = F.normalize(feat, dim=1)
        return logits, feat
# class SimpleConvNet(nn.Module):
#     def __init__(self, kernel_size=3, **kwargs):
#         super(SimpleConvNet, self).__init__()
#         padding = kernel_size // 2
#         layers = [
#             nn.Conv2d(1, 16, kernel_size=kernel_size, padding=padding),
#             nn.BatchNorm2d(16),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(16, 32, kernel_size=kernel_size, padding=padding),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 64, kernel_size=kernel_size, padding=padding),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#         ]
#         self.extracter = nn.Sequential(*layers)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.dim_in = 64
#         self.fc = nn.Linear(self.dim_in, 10)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#     def forward(self, x):
#         x = self.extracter(x)
#         x = self.avgpool(x)
#         feat = torch.flatten(x, 1)
#         logits = self.fc(feat)
#
#         return logits, feat
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dim_in = 128
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        feat = F.relu(x)
        x = self.dropout2(feat)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output, feat
class Simple3DNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dim_in = 128
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        feat = F.relu(x)
        x = self.dropout2(feat)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output, feat