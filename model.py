import torch
from transformers import BertModel
from torchvision.models import resnet50, ResNet50_Weights
from torch import nn

class CliPortModel(nn.Module):
    def __init__(self):
        super(CliPortModel, self).__init__()
        self.language_model = BertModel.from_pretrained('bert-base-uncased')
        self.visual_model = resnet50(weights=ResNet50_Weights.DEFAULT)

        for param in self.language_model.parameters():
            param.requires_grad = False
        for param in self.visual_model.parameters():
            param.requires_grad = False
        
        # Modify the last layer of ResNet to match BERT's feature size
        self.visual_model.fc = nn.Linear(self.visual_model.fc.in_features, self.language_model.config.hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(self.language_model.config.hidden_size * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # Output 4 coordinates (start and end)
        )

    def forward(self, images, texts):
        language_output = self.language_model(input_ids=texts['input_ids'], attention_mask=texts['attention_mask'])
        visual_output = self.visual_model(images.float())
        combined = torch.cat((language_output.pooler_output, visual_output), dim=1)
        coordinates = self.classifier(combined)
        return coordinates

if __name__ == '__main__':
    model = CliPortModel()
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    # print(model({'text': 'Pick up the red block', 'image': torch.rand((1, 3, 224, 224))}))