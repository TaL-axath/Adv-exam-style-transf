import os
import torchvision.transforms as transforms
from PIL import Image

# 图像加载，获取文件名
class ImageLoader:
    def __init__(self, folder_path):
        if not os.path.isdir(folder_path):
            raise ValueError(f"Folder '{folder_path}' does not exist.")
        self.folder_path = folder_path
        self.image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('jpg', 'jpeg', 'png'))])
        if not self.image_files:
            raise ValueError(f"No image files found in folder '{folder_path}'.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if idx >= len(self.image_files) or idx < 0:
            raise IndexError(f"Index {idx} out of range.")
        image_path = os.path.join(self.folder_path, self.image_files[idx])
        return Image.open(image_path)

    def get_filename(self, idx):
        if idx >= len(self.image_files) or idx < 0:
            raise IndexError(f"Index {idx} out of range.")
        return self.image_files[idx]

# 预处理图像
def load_image(image, max_size=400):
    image = image.convert('RGB')

    # 如果图片过大则调整大小
    size = min(max_size, max(image.size))

    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = transform(image).unsqueeze(0)  # 添加批次维度
    return image

# 保存图像
def save_image(tensor, path):
    image = tensor.clone().detach()
    image = image.squeeze(0)  # 去掉批次维度
    image = transforms.ToPILImage()(image)
    image.save(path)

if __name__ == '__main__':
    folder ='./content-image/'
    imageloader = ImageLoader(folder)
    print(f"文件夹中共有 {len(imageloader)} 张图片。")
    image = imageloader[0]
    image_name = imageloader.get_filename(0)
    image_name = image_name.split('.')[0]
    print(f"图片名称: {image_name}")
