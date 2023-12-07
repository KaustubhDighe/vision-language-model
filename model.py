import torch
from transformers import BertTokenizer, BertModel
from torchvision import models
from torch import nn

class CliPortModel(nn.Module):
    def __init__(self):
        super(CliPortModel, self).__init__()
        self.language_model = BertModel.from_pretrained('bert-base-uncased')
        self.visual_model = models.resnet50(pretrained=True)
        # Modify the last layer of ResNet to match BERT's feature size
        self.visual_model.fc = nn.Linear(self.visual_model.fc.in_features, self.language_model.config.hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(self.language_model.config.hidden_size * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # Assuming 2D manipulation actions
        )

    def forward(self, input_ids, attention_mask, visual_input):
        language_output = self.language_model(input_ids=input_ids, attention_mask=attention_mask)
        visual_output = self.visual_model(visual_input)
        combined = torch.cat((language_output.pooler_output, visual_output), dim=1)
        actions = self.classifier(combined)
        return actions

if __name__ == '__main__':
    # Example usage
    model = CliPortModel()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Dummy inputs
    text = "Pick up the red block"
    inputs = tokenizer(text, return_tensors="pt")
    visual_input = torch.rand((1, 3, 224, 224))  # Random image

    # Forward pass
    actions = model(inputs['input_ids'], inputs['attention_mask'], visual_input)
    print(actions)
