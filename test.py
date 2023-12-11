from generate_data import ShapesDataset
from model import StreamFCN
import torch
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw

def test(image_size=(224, 224), num_objects=2, num_samples=128, batch_size=16, device='cpu'):
    dataset = ShapesDataset(image_size, num_objects, num_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = StreamFCN(32)
    model.to(device)
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    # model.to(device)
    for data in dataloader:
        # images, texts, prompts, labels = data[0]['image'], data[0]['text'], data[0]['prompt'], data[1]
        # images, labels = images.to(device, dtype=torch.float32), labels.to(device, dtype=torch.float32)
        # texts['input_ids'], texts['attention_mask'] = texts['input_ids'].to(device, dtype=torch.long), texts['attention_mask'].to(device, dtype=torch.float32)
        # start_x,  start_y, end_x, end_y = model(images, texts)
        # outputs = torch.stack((start_x.argmax(dim=1), start_y.argmax(dim=1), end_x.argmax(dim=1), end_y.argmax(dim=1)), dim=1)
        # labels = labels.argmax(dim=2)
        # print(outputs, labels, prompts)

        outputs = model(data[0]['image'], data[0]['text'])
        labels = data[1]
        
        outputs = (outputs * image_size[0] + image_size[0] / 2)
        # end = end * image_size[0] + image_size[0] / 2
        labels = (labels * image_size[0] + image_size[0] / 2) 
        print(labels, outputs)
        for i in range(len(data[0]['image'])):
            img = Image.fromarray((data[0]['image'][i].permute(1, 2, 0).cpu().numpy()).astype('uint8'), mode='RGB')
            draw = ImageDraw.Draw(img)
            # draw.ellipse([start[i][0] - 2, start[i][1] - 2, start[i][0] + 2, start[i][1] + 2], fill=(255, 0, 0))
            # draw.ellipse([end[i][0] - 2, end[i][1] - 2, end[i][0] + 2, end[i][1] + 2], fill=(255, 0, 0))
            draw.ellipse([outputs[i][0] - 2, outputs[i][1] - 2, outputs[i][0] + 2, outputs[i][1] + 2], fill=(255, 0, 0))
            draw.ellipse([outputs[i][2] - 2, outputs[i][3] - 2, outputs[i][2] + 2, outputs[i][3] + 2], fill=(255, 0, 0))
            draw.ellipse([labels[i][0] - 2, labels[i][1] - 2, labels[i][0] + 2, labels[i][1] + 2], fill=(0, 0, 255))
            draw.ellipse([labels[i][2] - 2, labels[i][3] - 2, labels[i][2] + 2, labels[i][3] + 2], fill=(0, 0, 255))
            img.save(f'test/test{i}.png')
        # img = Image.fromarray((data['image'][0].permute(1, 2, 0).cpu().numpy()).astype('uint8'), mode='RGB')
        # draw = ImageDraw.Draw(img)
        # draw.ellipse([pick[0] - 2, pick[1] - 2, pick[0] + 2, pick[1] + 2], fill=(255, 0, 0))
        # draw.ellipse([place[0] - 2, place[1] - 2, place[0] + 2, place[1] + 2], fill=(255, 0, 0))
        # draw.ellipse([data['start'][0][0] - 2, data['start'][0][1] - 2, data['start'][0][0] + 2, data['start'][0][1] + 2], fill=(0, 0, 255))
        # draw.ellipse([data['end'][0][0] - 2, data['end'][0][1] - 2, data['end'][0][0] + 2, data['end'][0][1] + 2], fill=(0, 0, 255))
        # img.save(f'test/test.png')
        break


if __name__ == '__main__':
    device = 'cpu' #torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    test(device=device)