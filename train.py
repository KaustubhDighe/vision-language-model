from generate_data import ShapesDataset
from model import StreamFCN
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

def train(learning_rate=1e-4, num_epochs=60, batch_size=16, image_size=(224, 224), num_objects=2, num_samples=256, device='mps'):
    dataset = ShapesDataset(image_size, num_objects, num_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = StreamFCN(2)  # channel 0 = pick heatmap, channel 1 = place heatmap
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    # criterion = torch.nn.CrossEntropyLoss()

    losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for data in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
            # images, texts, labels = data[0]['image'], data[0]['text'], data[1]
            # images, labels = images.to(device, dtype=torch.float32), labels.to(device, dtype=torch.float32)
            # texts['input_ids'], texts['attention_mask'] = texts['input_ids'].to(device, dtype=torch.long), texts['attention_mask'].to(device, dtype=torch.float32)
            optimizer.zero_grad()
            # print(labels.shape)
            # print(data['text'])
            data[1] = data[1].to(device, dtype=torch.float32)
            data[0]['image'] = data[0]['image'].to(device, dtype=torch.float32)
            output = model(data[0]['image'], data[0]['text'])
            # print(start.shape, end.shape, labels.shape)
            
            # loss = criterion(outputs, labels)
            # second term is subtracted on purpose: it rewards the pick and place
            # predictions being far apart, discouraging pick == place collapse
            loss = criterion(output, data[1]) - criterion(output[:, 0:2], output[:, 2:4]) * 1e-3
            # loss = criterion(data['start'] / 224 - 0.5, pick / 224 - 0.5) + criterion(data['end'] / 224 - 0.5, place / 224 - 0.5)
            # loss.requires_grad = True
            loss.backward()
            
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader)}")
        losses.append(running_loss/len(dataloader))
    print("Training complete")
    plt.plot(losses)
    plt.savefig('loss.png')
    torch.save(model.state_dict(), 'model.pth')

if __name__ == '__main__':
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # train(num_epochs=5, device=device)
    train()
