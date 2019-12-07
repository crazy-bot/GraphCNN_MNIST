import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
# import torch_geometric.transforms as T
# # from torch_geometric.data import Data,DataLoader
# from torch_geometric.utils import normalized_cut
# from torch_geometric.nn import (NNConv, graclus, max_pool, max_pool_x,
#                                 global_mean_pool)

num_epochs = 100
batch_size = 128
learning_rate = 1e-3

MNIST_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
train_loader = torch.utils.data.DataLoader(
        MNIST('../data', train=True, download=True, transform= MNIST_transform),
        batch_size=64, shuffle=True)

test_loader = torch.utils.data.DataLoader( MNIST('../data', train=False, transform= MNIST_transform),
        batch_size=64, shuffle=False)

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.Tanh(),
            #nn.BatchNorm1d(),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.Tanh(),
            nn.MaxPool2d(2, stride=1),  # b, 8, 2, 2
            #nn.BatchNorm1d()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.Tanh(),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.Tanh(),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# class autoencoder(nn.Module):
#     def __init__(self):
#         super(autoencoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(28 * 28, 50),
#             nn.Tanh(),
#             nn.Linear(50, 50),
#             nn.ReLU(True), nn.Linear(64, 12), nn.ReLU(True), nn.Linear(12, 3))
#         self.decoder = nn.Sequential(
#             nn.Linear(3, 12),
#             nn.ReLU(True),
#             nn.Linear(12, 64),
#             nn.ReLU(True),
#             nn.Linear(64, 128),
#             nn.ReLU(True), nn.Linear(128, 28 * 28), nn.Tanh())
#
#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x

model = autoencoder().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-3)

for epoch in range(num_epochs):
    for data in train_loader:
        img, _ = data
        #img = img.view(img.size(0), -1)
        img = Variable(img).cuda()
        # ===================forward=====================
        output = model(img)
        loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch+1, num_epochs, loss.item()))
    if epoch % 10 == 0:
        torch.save(model.state_dict(), './conv_autoencoder.pth')
