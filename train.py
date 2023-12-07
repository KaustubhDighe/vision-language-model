from generate_data import ShapesDataset
from model import CliPortModel
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

def train(learning_rate=1e-4, num_epochs=10, batch_size=16, image_size=(224, 224), num_objects=3, num_samples=1024, device='cpu'):
    dataset = ShapesDataset(image_size, num_objects, num_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = CliPortModel()
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for data in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
            images, texts, labels = data[0]['image'], data[0]['text'], data[1]
            optimizer.zero_grad()
            
            outputs = model(images, texts)
            
            loss = criterion(outputs, labels.float())
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
    train(device=device)