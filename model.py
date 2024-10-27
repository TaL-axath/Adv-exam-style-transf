from torchvision.models import vgg19
import torch
import torch.nn as nn

# 定义VGG19模型，提取特定层的特征和最终分类结果
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.features = vgg19(pretrained=True).features.eval()
        self.avgpool = vgg19(pretrained=True).avgpool.eval()
        self.classifier = vgg19(pretrained=True).classifier.eval()

    def forward(self, x):
        features = []
        # 提取特定层的特征
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in {0, 5, 10, 19, 21}:  # 选择特定层的输出
                features.append(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        classification_result = self.classifier(x)

        return features, classification_result