import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

class info(Dataset):
    def __init__(self):
        super().__init__()
        self.xx = torch.tensor([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12]], dtype=torch.float32)
        self.yy = torch.tensor([[1],[4],[9],[16],[25],[36],[49],[64],[81],[100],[121],[144]], dtype=torch.float32)

    def __len__(self):
        return len(self.xx)
    
    def __getitem__(self, index):
        x = self.xx[index]
        y = self.yy[index]
        return x, y

dataset = info()
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.one = nn.Linear(1,8)
        self.two = nn.Linear(8,4)
        self.three = nn.Linear(4,1)
        self.rely = nn.ReLU()

    def forward(self, x):
        y = self.one(x)
        y = self.rely(y)
        y = self.two(y)
        y = self.rely(y)
        y = self.three(y)
        return y
    
model = NeuralNet()
epochs = 10000
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
pr = 0
for epoch in range(epochs):
    for x,y in dataloader:
        y_pred = model(x)
        fun_loss = nn.MSELoss()
        loss = fun_loss(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % 1000 == 0:
        pr += 10
        print(pr,'%')

for i in range(100):
    a = float(input('Print your num: '))
    a = torch.tensor([[a]], dtype=torch.float32)

    with torch.no_grad():
        print('Maybe:', model(a).item())

        
        

        
        
