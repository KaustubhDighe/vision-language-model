import torch
from transformers import BertModel
from torchvision.models import resnet50, ResNet50_Weights
from torch import nn

class BoundingBoxModel(nn.Module):
    def __init__(self, num_features, num_boxes):
        super(BoundingBoxModel, self).__init__()
        self.linear1 = nn.Linear(num_features, 512)
        self.fc = nn.Linear(512, num_boxes)
    
    def forward(self, x):
        x = nn.Dropout(0.1)(x)
        x = nn.Sigmoid()(self.linear1(x))
        x = nn.Dropout(0.1)(x)
        x = self.fc(x)
        return nn.Softmax(dim=1)(x)

class CliPortModel(nn.Module):
    def __init__(self):
        super(CliPortModel, self).__init__()
        self.language_model = BertModel.from_pretrained('bert-base-uncased')
        self.visual_model = resnet50(weights=ResNet50_Weights.DEFAULT)

        for param in self.language_model.parameters():
            param.requires_grad = False
        for param in self.visual_model.parameters():
           param.requires_grad = False

        # self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        
        # Modify the last layer of ResNet to match BERT's feature size
        self.visual_model.fc = nn.Linear(self.visual_model.fc.in_features, self.language_model.config.hidden_size)
        self.start_x = BoundingBoxModel(self.language_model.config.hidden_size * 2, 4)
        self.start_y = BoundingBoxModel(self.language_model.config.hidden_size * 2, 4)
        self.end_x = BoundingBoxModel(self.language_model.config.hidden_size * 2, 4)
        self.end_y =BoundingBoxModel(self.language_model.config.hidden_size * 2, 4)
        # self.fusion = nn.Sequential(
        #     nn.Linear(self.language_model.config.hidden_size * 2, 256),
        #     nn.Dropout(0.3),
        #     nn.ReLU(),
        #     nn.Linear(256, 4),  # Predict start coordinates (x0, y0)
        # )

    def forward(self, images, texts):
        language_output = self.language_model(input_ids=texts['input_ids'], attention_mask=texts['attention_mask'])
        visual_output = self.visual_model(images.float())
        combined = torch.cat((language_output.pooler_output, visual_output), dim=1)
        start_x = torch.softmax(self.start_x(combined), dim=1)
        start_y = torch.softmax(self.start_y(combined), dim=1)
        end_x = torch.softmax(self.end_x(combined), dim=1)
        end_y = torch.softmax(self.end_y(combined), dim=1)
        return start_x, start_y, end_x, end_y

if __name__ == '__main__':
    model = CliPortModel()
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    # print(model({'text': 'Pick up the red block', 'image': torch.rand((1, 3, 224, 224))}))