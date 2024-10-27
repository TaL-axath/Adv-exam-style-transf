import torch
import torch.nn as nn

# 内容损失函数
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        return nn.functional.mse_loss(input, self.target)


# 风格损失函数
class StyleLoss(nn.Module):
    def __init__(self, target):
        super(StyleLoss, self).__init__()
        self.target = self.gram_matrix(target).detach()

    def gram_matrix(self, input):
        batch_size, channels, height, width = input.size()
        features = input.view(batch_size * channels, height * width)
        G = torch.mm(features, features.t())
        return G.div(batch_size * channels * height * width)

    def forward(self, input):
        G = self.gram_matrix(input)
        return nn.functional.mse_loss(G, self.target)