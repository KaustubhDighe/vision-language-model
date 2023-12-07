import torch
from transformers import BertTokenizer, BertModel
from torchvision import models
from torch import nn

class CliPortModel(nn.Module):
    def __init__(self):
        super(CliPortModel, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.language_model = BertModel.from_pretrained('bert-base-uncased')
        self.visual_model = models.resnet50(pretrained=True)
        
        # Modify the last layer of ResNet to match BERT's feature size
        self.visual_model.fc = nn.Linear(self.visual_model.fc.in_features, self.language_model.config.hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(self.language_model.config.hidden_size * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # Output 4 coordinates (start and end)
        )

    def forward(self, inputs):
        tokenized_input = self.tokenizer(
                            inputs['text'], 
                            return_tensors="pt", 
                            padding=True, 
                            truncation=True, 
                            max_length=512
                            )
        language_output = self.language_model(
                            input_ids=tokenized_input['input_ids'], 
                            attention_mask=tokenized_input['attention_mask']
                            )
        visual_output = self.visual_model(inputs['image'].float())
        combined = torch.cat((language_output.pooler_output, visual_output), dim=1)
        coordinates = self.classifier(combined)
        return coordinates

if __name__ == '__main__':
    model = CliPortModel()
    print(model({'text': 'Pick up the red block', 'image': torch.rand((1, 3, 224, 224))}))