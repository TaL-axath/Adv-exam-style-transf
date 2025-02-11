import torch
import torch.optim as optim
from model import VGG
from loss import ContentLoss, StyleLoss
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn


def attack(content_img, style_img, num_steps=500, target_class=498, style_weight=1e8, content_weight=2, class_weight=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    content_img = content_img.to(device)
    style_img = style_img.to(device)

    model = VGG().to(device)
    target_label = torch.tensor([target_class], dtype=torch.long).to(device)
    criterion = nn.CrossEntropyLoss()

    # 提取风格和内容特征
    style_features, _ = model(style_img)
    content_features, content_classification = model(content_img)

    # 初始化输入图像（使用内容图像作为初始图像）
    input_img = content_img.clone().requires_grad_(True).to(device)

    # 定义优化器
    optimizer = optim.LBFGS([input_img])

    style_losses = []
    content_losses = []

    # 创建损失模块
    for sf, cf in zip(style_features, content_features):
        content_losses.append(ContentLoss(cf))
        style_losses.append(StyleLoss(sf))

    run = [0]
    while run[0] <= num_steps:

        def closure():
            optimizer.zero_grad()

            input_features, input_classification = model(input_img)
            content_loss = 0
            style_loss = 0

            for cl, input_f in zip(content_losses, input_features):
                content_loss += content_weight * cl(input_f)

            for sl, input_f in zip(style_losses, input_features):
                style_loss += style_weight * sl(input_f)

            classfication_loss = criterion(input_classification, target_label)
            classfication_loss = class_weight * classfication_loss

            loss = content_loss + style_loss + classfication_loss
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print(f'Step {run[0]}, Content Loss: {content_loss.item():4f}, Style Loss: {style_loss.item():4f}, Classification Loss: {classfication_loss.item():4f}')
                input_classification_softmax = F.softmax(input_classification, dim=1)
                target_class_probability = input_classification_softmax[:, target_class]
                print(f'Probability of Classification {target_class}: {target_class_probability.item():4f}')
            return loss

        optimizer.step(closure)

    _, input_classification = model(input_img)
    input_classification_softmax = F.softmax(input_classification, dim=1)
    target_class_probability = input_classification_softmax[:, target_class]
    # 取消归一化并返回结果
    unnormalize = transforms.Normalize(
        mean=[-2.118, -2.036, -1.804],
        std=[4.367, 4.464, 4.444]
    )
    result = unnormalize(input_img)
    return result, target_class_probability