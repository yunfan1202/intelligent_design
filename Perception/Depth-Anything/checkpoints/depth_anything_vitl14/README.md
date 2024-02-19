---
tags:
- depth_anything
- depth-estimation
---

# Depth Anything model, large

The model card for our paper [Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data](https://arxiv.org/abs/2401.10891).

You may also try our [demo](https://huggingface.co/spaces/LiheYoung/Depth-Anything) and visit our [project page](https://depth-anything.github.io/).

## Installation

First, install the Depth Anything package:
```
git clone https://github.com/LiheYoung/Depth-Anything
cd Depth-Anything
pip install -r requirements.txt
```

## Usage

Here's how to run the model:

```python
import numpy as np
from PIL import Image
import cv2
import torch

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose

model = DepthAnything.from_pretrained("LiheYoung/depth_anything_vitl14")

transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

image = Image.open("...")
image = np.array(image) / 255.0
image = transform({'image': image})['image']
image = torch.from_numpy(image).unsqueeze(0)

depth = model(image)
```