import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def point_as_prompt():
    # choose a point as the prompt
    input_point = np.array([[500, 375]])
    input_label = np.array([1])

    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_points(input_point, input_label, plt.gca())
    plt.axis('on')
    plt.show()

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    print(masks.shape)

    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()
    # --------------------------------------------------------------------
    # Specifying a specific object with additional points
    # input_point = np.array([[500, 375], [1125, 625]])
    # input_label = np.array([1, 1])
    # --------------------------------------------------------------------
    # To exclude the car and specify just the window, a background point (with label 0, here shown in red) can be supplied.
    input_point = np.array([[500, 375], [1125, 625]])
    input_label = np.array([1, 0])

    mask_input = logits[np.argmax(scores), :, :]
    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        mask_input=mask_input[None, :, :],
        multimask_output=False,
    )
    print(masks.shape)
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(masks, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.axis('off')
    plt.show()


def box_as_prompt():
    # The model can also take a box as input, provided in xyxy format.
    input_box = np.array([425, 600, 700, 875])
    # print(input_box[None, :])
    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
    )
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(masks[0], plt.gca())
    show_box(input_box, plt.gca())
    plt.axis('off')
    plt.show()

    # --------------------------------------------------------------------
    # Points and boxes may be combined, just by including both types of prompts to the predictor. Here this can be used to select just the trucks's tire, instead of the entire wheel.
    input_box = np.array([425, 600, 700, 875])
    input_point = np.array([[575, 750]])
    input_label = np.array([0])
    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        box=input_box,
        multimask_output=False,
    )
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(masks[0], plt.gca())
    show_box(input_box, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.axis('off')
    plt.show()

    # --------------------------------------------------------------------
    # SamPredictor can take multiple input prompts for the same image
    input_boxes = torch.tensor([
        [75, 275, 1725, 850],
        [425, 600, 700, 875],
        [1375, 550, 1650, 800],
        [1240, 675, 1400, 750],
    ], device=predictor.device)
    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    print(masks.shape)
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    for box in input_boxes:
        show_box(box.cpu().numpy(), plt.gca())
    plt.axis('off')
    plt.show()


def batched_inference():
    # If all prompts are available in advance, it is possible to run SAM directly in an end-to-end fashion. This also allows batching over images.
    image1 = image  # truck.jpg from above
    image1_boxes = torch.tensor([
        [75, 275, 1725, 850],
        [425, 600, 700, 875],
        [1375, 550, 1650, 800],
        [1240, 675, 1400, 750],
    ], device=sam.device)

    image2 = cv2.imread('images/groceries.jpg')
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    image2_boxes = torch.tensor([
        [450, 170, 520, 350],
        [350, 190, 450, 350],
        [500, 170, 580, 350],
        [580, 170, 640, 350],
    ], device=sam.device)
    from segment_anything.utils.transforms import ResizeLongestSide
    resize_transform = ResizeLongestSide(sam.image_encoder.img_size)

    def prepare_image(image, transform, device):
        image = transform.apply_image(image)
        image = torch.as_tensor(image, device=device.device)
        return image.permute(2, 0, 1).contiguous()

    batched_input = [
        {
            'image': prepare_image(image1, resize_transform, sam),
            'boxes': resize_transform.apply_boxes_torch(image1_boxes, image1.shape[:2]),
            'original_size': image1.shape[:2]
        },
        {
            'image': prepare_image(image2, resize_transform, sam),
            'boxes': resize_transform.apply_boxes_torch(image2_boxes, image2.shape[:2]),
            'original_size': image2.shape[:2]
        }
    ]
    batched_output = sam(batched_input, multimask_output=False)
    print(batched_output[0].keys())

    fig, ax = plt.subplots(1, 2, figsize=(20, 20))

    ax[0].imshow(image1)
    for mask in batched_output[0]['masks']:
        show_mask(mask.cpu().numpy(), ax[0], random_color=True)
    for box in image1_boxes:
        show_box(box.cpu().numpy(), ax[0])
    ax[0].axis('off')

    ax[1].imshow(image2)
    for mask in batched_output[1]['masks']:
        show_mask(mask.cpu().numpy(), ax[1], random_color=True)
    for box in image2_boxes:
        show_box(box.cpu().numpy(), ax[1])
    ax[1].axis('off')

    plt.tight_layout()
    plt.show()


image = cv2.imread('images/truck.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

sam_checkpoint = "../checkpoints/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)
predictor.set_image(image)

# 可以分别都试试
point_as_prompt()
box_as_prompt()
# batched_inference()