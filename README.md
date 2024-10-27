Reference:
https://github.com/Yolumia/Image_style_transfer_base_vgg19/

## How to run:
```
python main.py
```

The content image and style image are saved in `content-image` and `style-image`

The result is saved in `output-image`

The name of the adv example is `{content-image name}_{style-image name}_{probability of target class}.jpg`

On the default setting, the target class is 498(n03032252 cinema, movie theater, movie theatre, movie house, picture palace)

To set the hyper-parameters, please modify the parameters in `attack.py`

## Method

The model is modified based on VGG19, it returns the features together with the classification result

Use optimizer LBFGS to generate the adv example, iterates by loss.backward()

The loss function is defined in `loss.py`, they are called to calculate loss

