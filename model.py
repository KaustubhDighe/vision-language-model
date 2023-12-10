# import torch
# from transformers import CLIPModel, AutoProcessor
# from PIL import Image
# import requests

# model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
# processor = AutoProcessor.from_pretrained('openai/clip-vit-base-patch32')
# model.eval()

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

# inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)
# outputs = model(**inputs)
# print(outputs.image_embeds.shape)
# logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
# probs = logits_per_image.softmax(dim=1)
# print("Label probs:", probs)
# print("Label probs:", probs.tolist())

import torch
import clip
from PIL import Image
import requests
from torchsummary import summary

# device = "mps"
model, preprocess = clip.load("ViT-B/32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = preprocess(Image.open(requests.get(url, stream=True).raw)).unsqueeze(0)

text = clip.tokenize(["a diagram", "a dog", "a cat"])

summary(model.encode_image, (3, 224, 224))
    # text_features = model.encode_text(text)
    
    # logits_per_image, logits_per_text = model(image, text)
    # probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# with torch.no_grad():
#     image_features = model.encode_image(image)
#     text_features = model.encode_text(text)
    
#     logits_per_image, logits_per_text = model(image, text)
#     probs = logits_per_image.softmax(dim=-1).cpu().numpy()