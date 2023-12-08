from PIL import Image, ImageDraw
import random
import numpy as np
import torch
from torchvision import transforms
from transformers import BertTokenizer

class ShapesDataset(torch.utils.data.Dataset):
    def __init__(self, image_size, num_objects, num_samples, show=False):
        self.colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'pink']
        self.rgb = {
            'red' : (255, 0, 0),
            'green' : (0, 255, 0),
            'blue' : (0, 0, 255),
            'yellow' : (255, 255 ,0),
            'purple' : (128, 0, 128),
            'orange' : (255, 165, 0),
            'pink' : (255, 192, 203) 
        }
        self.shapes = ['square', 'square'] # , 'triangle', 'hexagon']
        self.image_size = image_size
        self.num_objects = num_objects
        self.num_samples = num_samples
        self.N = 4
        self.shape_size = image_size[0] // self.N
        self.show = show
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    def __len__(self):
        return self.num_samples
    
    def draw_shape(self, draw, color, shape, bbox):
        i, j = bbox
        x0, y0, x1, y1 = i * self.shape_size, j * self.shape_size, (i + 1) * self.shape_size, (j + 1) * self.shape_size
        if shape == 'square':
            # d = np.random.randint(0, self.shape_size // 2)
            d = 0
            draw.polygon([(x0 + d, y0), (x1, y0 + d), (x1-d, y1), (x0, y1-d)], fill=self.rgb[color])
        elif shape == 'circle':
            # d = np.random.randint(0, self.shape_size // 4)
            d = 0
            draw.ellipse([x0 + d, y0 + d, x1 - d, y1 - d], fill=self.rgb[color])
        elif shape == 'triangle':
            A = (np.random.randint(x0, x1), y0)
            B = (x1, np.random.randint(y0, y1))
            C = (np.random.randint(x0, x1), y1)
            draw.polygon([A, B, C], fill=self.rgb[color])
        elif shape == 'hexagon':
            A = (np.random.randint(x0, x1), y0)
            B = (x1, np.random.randint(y0, y1 - 1))
            C = (x1, np.random.randint(B[1], y1))
            D = (np.random.randint(x0, x1), y1)
            E = (x0, np.random.randint(y0 + 1, y1))
            F = (x0, np.random.randint(y0, E[1]))
            draw.polygon([A, B, C, D, E, F] , fill=self.rgb[color])
    
    def __getitem__(self, idx):
        img = Image.new('RGB', self.image_size, color='black')
        draw = ImageDraw.Draw(img)

        colors = random.sample(self.colors, self.num_objects)
        shapes = random.sample(self.shapes, self.num_objects)
        bboxes = random.sample([(i, j) for i in range(self.N) for j in range(self.N)], self.num_objects)

        for i in range(self.num_objects):
            self.draw_shape(draw, colors[i], shapes[i], bboxes[i])
        
        if self.show:
            img.show()

        def get_loc_from_bbox(bbox):
            return ((np.array(list(bbox)) + 0.5) * self.shape_size - self.image_size[0] / 2) / self.image_size[0]
        
        transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
                    ])
        
        text = f"from {colors[0]} square to the {colors[1]} square"
        tokenized_text = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        tokenized_text = {'input_ids': tokenized_text['input_ids'][0], 'attention_mask': tokenized_text['attention_mask'][0]}
        # start, end = get_loc_from_bbox(bboxes[0]), get_loc_from_bbox(bboxes[1])
        # start, end = bbox[0], bbox[1]
        trajectory = np.zeros((4, self.N))
        trajectory[0, bboxes[0][0]] = 1
        trajectory[1, bboxes[0][1]] = 1
        trajectory[2, bboxes[1][0]] = 1
        trajectory[3, bboxes[1][1]] = 1
        
        return ({'image' : transform(img), 'text': tokenized_text, 'prompt': text, 'start': bboxes[0], 'end': bboxes[1]}, 
                torch.tensor(trajectory, dtype=torch.float32))

if __name__ == '__main__':
    # Generate an image and show it
    dataset = ShapesDataset((224, 224), 3, 1024, show=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    print(next(iter(dataloader)))