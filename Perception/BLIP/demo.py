from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

import requests


def load_image(image_size, path, device):
    raw_image = Image.open(path).convert('RGB')
    # w, h = raw_image.size
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    image = transform(raw_image).unsqueeze(0).to(device)
    return image


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
img_path = "demo.jpg"
# img_path = "../CLIP/butterfly.jpg"


def image_caption():
    print("-----------------------image_caption-----------------------")
    from models.blip import blip_decoder
    image_size = 384
    image = load_image(image_size, img_path, device=device)
    print(image.shape)

    model_path = 'checkpoints/model_base_capfilt_large.pth'
    model = blip_decoder(pretrained=model_path, image_size=image_size, vit='base')
    model.eval()
    model = model.to(device)

    with torch.no_grad():
        # beam search
        caption = model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5)
        # nucleus sampling
        # caption = model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5)
        print('caption: ' + caption[0])


def VQA():
    print("-----------------------VQA-----------------------")
    from models.blip_vqa import blip_vqa

    image_size = 480
    image = load_image(image_size, img_path, device=device)

    model_url = 'checkpoints/model_base_vqa_capfilt_large.pth'

    model = blip_vqa(pretrained=model_url, image_size=image_size, vit='base')
    model.eval()
    model = model.to(device)

    question = 'where is the woman sitting?'

    with torch.no_grad():
        answer = model(image, question, train=False, inference='generate')
        print('question: where is the woman sitting?')
        print('answer: ' + answer[0])


def feature_extraction():
    print("-----------------------feature_extraction-----------------------")
    from models.blip import blip_feature_extractor
    import torch.nn.functional as F
    image_size = 224
    image = load_image(image_size, img_path, device=device)

    model_url = 'checkpoints/model_base.pth'

    model = blip_feature_extractor(pretrained=model_url, image_size=image_size, vit='base')
    model.eval()
    model = model.to(device)

    caption = 'a woman sitting on the beach with a dog'

    # multimodal_feature = model(image, caption, mode='multimodal')[0, 0] # torch.Size([768])
    image_feature = model(image, caption, mode='image')[0, 0] # torch.Size([768])
    text_feature = model(image, caption, mode='text')[0, 0] # torch.Size([768])
    print(image_feature.shape, text_feature.shape)

def image_text_matching():
    print("-----------------------image_text_matching-----------------------")
    from models.blip_itm import blip_itm
    image_size = 384
    image = load_image(image_size, img_path, device=device)

    model_url = 'checkpoints/model_base_retrieval_coco.pth'

    model = blip_itm(pretrained=model_url, image_size=image_size, vit='base')
    model.eval()
    model = model.to(device=device)

    caption = 'a woman sitting on the beach with a dog'
    print('text: %s' % caption)

    itm_output = model(image, caption, match_head='itm')
    itm_score = torch.nn.functional.softmax(itm_output, dim=1)[:, 1]
    print('The image and text is matched with a probability of %.4f' % itm_score)

    itc_score = model(image, caption, match_head='itc')
    print('The image feature and text feature has a cosine similarity of %.4f' % itc_score)

image_caption()
VQA()
feature_extraction()
image_text_matching()
