import os
import torch
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw

from generate_data import ShapesDataset
from model import StreamFCN

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def unnormalize_to_pil(img_tensor):
    # reverse the ToTensor + Normalize applied in generate_data.py so pixels
    # are viewable RGB again instead of mean/std-shifted floats
    img = img_tensor.detach().cpu() * STD + MEAN
    img = (img.clamp(0, 1) * 255).permute(1, 2, 0).numpy().astype('uint8')
    return Image.fromarray(img, mode='RGB')


def test(image_size=(224, 224), num_objects=2, num_samples=128, batch_size=16, device='cpu'):
    os.makedirs('test', exist_ok=True)

    dataset = ShapesDataset(image_size, num_objects, num_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = StreamFCN(2, device=device)  # channel 0 = pick heatmap, channel 1 = place heatmap
    model.to(device)
    model.load_state_dict(torch.load('model.pth', map_location=device))
    model.eval()

    with torch.no_grad():
        data = next(iter(dataloader))
        images = data[0]['image'].to(device, dtype=torch.float32)
        texts = data[0]['text']
        labels = data[1]

        outputs = model(images, texts)
        outputs = outputs * image_size[0] + image_size[0] / 2
        labels = labels * image_size[0] + image_size[0] / 2

        for i in range(len(images)):
            print(f"[{i}] \"{texts[i]}\"  pred={outputs[i].tolist()}  label={labels[i].tolist()}")

            img = unnormalize_to_pil(images[i])
            draw = ImageDraw.Draw(img)
            # predicted pick (square) / place (circle) in red
            draw.rectangle([outputs[i][0] - 4, outputs[i][1] - 4, outputs[i][0] + 4, outputs[i][1] + 4], fill=(255, 0, 0))
            draw.ellipse([outputs[i][2] - 4, outputs[i][3] - 4, outputs[i][2] + 4, outputs[i][3] + 4], fill=(255, 0, 0))
            # ground-truth pick (square) / place (circle) in blue
            draw.rectangle([labels[i][0] - 4, labels[i][1] - 4, labels[i][0] + 4, labels[i][1] + 4], fill=(0, 0, 255))
            draw.ellipse([labels[i][2] - 4, labels[i][3] - 4, labels[i][2] + 4, labels[i][3] + 4], fill=(0, 0, 255))
            img.save(f'test/test{i}.png')

    print(f"Saved {len(images)} annotated test images to test/ (red = predicted, blue = ground truth)")


if __name__ == '__main__':
    device = 'cpu'
    test(device=device)
