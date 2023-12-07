from PIL import Image, ImageDraw
import random
import numpy as np
import torch

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
        self.shapes = ['square', 'circle', 'triangle', 'hexagon']
        self.image_size = image_size
        self.num_objects = num_objects
        self.num_samples = num_samples
        self.shape_size = 100
        self.N = self.image_size[0] // self.shape_size
        self.show = show
    
    def __len__(self):
        return self.num_samples
    
    def draw_shape(self, draw, color, shape, bbox):
        i, j = bbox
        x0, y0, x1, y1 = i * self.shape_size, j * self.shape_size, (i + 1) * self.shape_size, (j + 1) * self.shape_size
        if shape == 'square':
            d = np.random.randint(0, self.shape_size // 2)
            draw.polygon([(x0 + d, y0), (x1, y0 + d), (x1-d, y1), (x0, y1-d)], fill=self.rgb[color])
        elif shape == 'circle':
            d = np.random.randint(0, self.shape_size // 4)
            draw.ellipse([x0 + d, y0 + d, x1 - d, y1 - d], fill=self.rgb[color])
        elif shape == 'triangle':
            A = (np.random.randint(x0, x1), y0)
            B = (x1, np.random.randint(y0, y1))
            C = (np.random.randint(x0, x1), y1)
            draw.polygon([A, B, C], fill=self.rgb[color])
        elif shape == 'hexagon':
            A = (np.random.randint(x0, x1), y0)
            B = (x1, np.random.randint(y0, y1))
            C = (x1, np.random.randint(B[1], y1))
            D = (np.random.randint(x0, x1), y1)
            E = (x0, np.random.randint(y0, y1))
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
            return (np.array(list(bbox)) + 0.5) * self.shape_size
        
        text = f"pick the {colors[0]} block and place it over the {colors[1]} block"
        start, end = get_loc_from_bbox(bboxes[0]), get_loc_from_bbox(bboxes[1])
        return ({'image' : np.array(img), 'text': text}, {'start': start, 'end': end})

if __name__ == '__main__':
    # Generate an image and show it
    dataset = ShapesDataset((500, 500), 3, 1024, show=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    print(next(iter(dataloader)).shape)