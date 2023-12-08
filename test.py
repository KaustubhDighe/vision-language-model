from generate_data import ShapesDataset
from model import CliPortModel
import torch
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw

def test(image_size=(224, 224), num_objects=2, num_samples=1024, batch_size=16, device='cpu'):
    dataset = ShapesDataset(image_size, num_objects, num_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = CliPortModel()
    model.to(device)
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    # model.to(device)
    for data in dataloader:
        images, texts, prompts, labels = data[0]['image'], data[0]['text'], data[0]['prompt'], data[1]
        images, labels = images.to(device, dtype=torch.float32), labels.to(device, dtype=torch.float32)
        texts['input_ids'], texts['attention_mask'] = texts['input_ids'].to(device, dtype=torch.long), texts['attention_mask'].to(device, dtype=torch.float32)
        start_x,  start_y, end_x, end_y = model(images, texts)
        outputs = torch.stack((start_x.argmax(dim=1), start_y.argmax(dim=1), end_x.argmax(dim=1), end_y.argmax(dim=1)), dim=1)
        labels = labels.argmax(dim=2)
        print(outputs, labels, prompts)
        
        outputs = (outputs * image_size[0] + image_size[0] / 2) / 4
        # end = end * image_size[0] + image_size[0] / 2
        labels = (labels * image_size[0] + image_size[0] / 2) / 4
        for i in range(len(images)):
            img = Image.fromarray((images[i].permute(1, 2, 0).cpu().numpy()).astype('uint8'), mode='RGB')
            draw = ImageDraw.Draw(img)
            # draw.ellipse([start[i][0] - 2, start[i][1] - 2, start[i][0] + 2, start[i][1] + 2], fill=(255, 0, 0))
            # draw.ellipse([end[i][0] - 2, end[i][1] - 2, end[i][0] + 2, end[i][1] + 2], fill=(255, 0, 0))
            draw.ellipse([outputs[i][0] - 2, outputs[i][1] - 2, outputs[i][0] + 2, outputs[i][1] + 2], fill=(255, 0, 0))
            draw.ellipse([outputs[i][2] - 2, outputs[i][3] - 2, outputs[i][2] + 2, outputs[i][3] + 2], fill=(255, 0, 0))
            draw.ellipse([labels[i][0] - 2, labels[i][1] - 2, labels[i][0] + 2, labels[i][1] + 2], fill=(0, 0, 255))
            draw.ellipse([labels[i][2] - 2, labels[i][3] - 2, labels[i][2] + 2, labels[i][3] + 2], fill=(0, 0, 255))
            img.save(f'test/test{i}.png')
        break

if __name__ == '__main__':
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    test(device=device)