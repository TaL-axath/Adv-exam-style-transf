import os
from utils import ImageLoader, load_image, save_image
from attack import attack

if __name__ == '__main__':
    # 指定模型存储目录
    os.environ['TORCH_HOME'] = './model'

    content_folder = './content-image/'
    style_folder = './style-image/'
    content_imageloader = ImageLoader(content_folder)
    style_imageloader = ImageLoader(style_folder)
    print(f"content图片共 {len(content_imageloader)} 张")
    print(f"style图片共 {len(style_imageloader)} 张")

    for i in range(len(content_imageloader)):
        image = content_imageloader[i]
        image_name = content_imageloader.get_filename(i)
        image_name = image_name.split('.')[0]
        for j in range(len(style_imageloader)):
            style_image = style_imageloader[j]
            style_image_name = style_imageloader.get_filename(j)
            style_image_name = style_image_name.split('.')[0]
            print(f"content图片名称: {image_name}")
            print(f"style图片名称: {style_image_name}")
            content_img = load_image(image)
            style_img = load_image(style_image)
            result, target_class_probability = attack(content_img, style_img)
            target_class_probability = target_class_probability.cpu().detach().numpy()
            print(f"对抗样本的概率为 {target_class_probability}")
            output_image_path = f'./output-image-test/{image_name}_{style_image_name}_{target_class_probability}.jpg'
            save_image(result, output_image_path)
            print(f"对抗样本生成，图像已保存为 {output_image_path}")
