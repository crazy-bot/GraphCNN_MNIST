import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torch_geometric.data import Data,Batch
from torch_geometric.nn import GCNConv, ChebConv, global_mean_pool
from torch_geometric.nn import (NNConv, graclus, max_pool, max_pool_x,
                                global_mean_pool)
from torch_geometric.utils import normalized_cut
import numpy as np
import itertools
import matplotlib.pyplot as plt
import argparse
import torch.nn.functional as F

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
        #x = self.decoder(x)
        return x


MNIST_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
train_loader = torch.utils.data.DataLoader(
        MNIST('../data', train=True, download=True, transform= MNIST_transform),
        batch_size=64, shuffle=True)

test_loader = torch.utils.data.DataLoader( MNIST('../data', train=False, transform= MNIST_transform),
        batch_size=64, shuffle=False)
FeatModel = autoencoder().cuda()
FeatModel.load_state_dict(torch.load('conv_autoencoder.pth'))
FeatModel.eval()

def buildGraph(feat,label):
    B = feat.shape[0]
    NoOfNodes = feat.shape[1]
    #feat.reshape(B,NoOfNodes,-1)
    edge_index = list(itertools.permutations(np.arange(0,NoOfNodes),2))
    edge_index = torch.LongTensor(edge_index).T
    listofData = []
    for i in range(0,B):
        feat_arr = feat[i].detach().cpu().numpy().reshape(NoOfNodes,-1)
        edge_attr = np.asarray([np.linalg.norm(a-b) for a,b in itertools.product(feat_arr,feat_arr)])
        # for a in feat_arr[i]:
        #     for b in feat_arr[i]:
        #         print(np.linalg.norm(a-b))
        edge_attr = edge_attr [edge_attr > 0]
        edge_attr = torch.Tensor(edge_attr).view(-1)
        data = Data(x=torch.Tensor(feat_arr), edge_index=edge_index, edge_attr=edge_attr, y=label[i].view(-1))
        listofData.append(data)
    batch = Batch().from_data_list(listofData)

    return batch

# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         # self.conv1 = GCNConv(dataset.num_features, 16, cached=True,
#         #                      normalize=not args.use_gdc)
#         # self.conv2 = GCNConv(16, dataset.num_classes, cached=True,
#         #                      normalize=not args.use_gdc)
#         self.conv1 = ChebConv(4, 16, K=4)
#         self.conv2 = ChebConv(16, 10, K=4)
#         self.net = nn.Sequential(self.conv1,nn.ReLU(),self.conv2,nn.ReLU(),nn.LogSoftmax())
#
#         # self.reg_params = self.conv1.parameters()
#         # self.non_reg_params = self.conv2.parameters()
#
#     def forward(self,data):
#         x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
#         x = F.relu(self.conv1(x, edge_index, edge_weight))
#         x = F.dropout(x, training=self.training)
#         x = self.conv2(x, edge_index, edge_weight)
#         x = F.log_softmax(x, dim=1)
#         # batch_idx = np.arange(64)
#         # batch = torch.LongTensor(np.repeat(batch_idx,8)).cuda()
#         batch_out = global_mean_pool(x,data.batch)
#         return batch_out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        nn1 = nn.Sequential(nn.Linear(2, 25), nn.ReLU(), nn.Linear(25, 32))
        self.conv1 = NNConv(d.num_features, 32, nn1, aggr='mean')

        nn2 = nn.Sequential(nn.Linear(2, 25), nn.ReLU(), nn.Linear(25, 2048))
        self.conv2 = NNConv(32, 64, nn2, aggr='mean')

        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, d.num_classes)

    def forward(self, data):
        data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        data.edge_attr = None
        data = max_pool(cluster, data, transform=transform)

        data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        x, batch = max_pool_x(cluster, data.x, data.batch)

        x = global_mean_pool(x, batch)
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return F.log_softmax(self.fc2(x), dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GraphModel = Net().to(device)
optimizer = torch.optim.Adam(GraphModel.parameters(), lr=0.01)

def train(args,epoch):
    GraphModel.train()
    train_loss = 0
    correct = 0
    for batch_idx, data in enumerate(train_loader):
        img, label = data
        img = Variable(img).cuda()
        # ===================forward=====================
        feat = FeatModel(img)
        # print('feat:',feat)
        # build graph from features
        graphBatch = buildGraph(feat, label)
        graphBatch = graphBatch.to(device)
        optimizer.zero_grad()
        output = GraphModel(graphBatch)
        loss = F.nll_loss(output, graphBatch.y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(graphBatch.y).sum().item()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), loss.item()))
            # print('Train Epoch: {} [batch_idx:{} datalen: {}/{}]\tLoss: {:.6f}'.format(
            #     epoch, batch_idx * len(data.y), len(train_loader.dataset), loss.item()))

    train_loss = train_loss / (batch_idx + 1)
    torch.save(GraphModel.state_dict(), 'Graphmodel.pt')
    print(
        '\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            train_loss, correct, len(train_loader.dataset),
            100. * correct / len(train_loader.dataset)))
    return train_loss, correct


def test():
    GraphModel.eval()
    correct = 0
    test_loss = 0

    with torch.no_grad():
        for data in test_loader:
            img, label = data
            img = Variable(img).cuda()
            # ===================forward=====================
            feat = FeatModel(img)
            # build graph from features
            graphBatch = buildGraph(feat, label)
            graphBatch = graphBatch.to(device)
            output = GraphModel(graphBatch)
            test_loss += F.nll_loss(output, graphBatch.y,reduction='sum').item()
            pred = output.argmax(dim=1)
            correct += pred.eq(graphBatch.y).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))
    return test_loss,correct


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch GraphCNN MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64,
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=200,
                        help='how many batches to wait before logging training status')

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    epoch_train_loss = []
    epoch_test_loss = []
    epoch_train_acc = []
    epoch_test_acc = []
    for epoch in range(1, args.epochs + 1):
        train_loss,train_correct = train(args, epoch)
        test_loss,test_correct = test()
        epoch_train_loss.append(train_loss)
        epoch_test_loss.append(test_loss)
        epoch_train_acc.append(train_correct)
        epoch_test_acc.append(test_correct)
    #################plot loss and accuracy graph
    fig, ax = plt.subplots()
    ax.plot(np.arange(epoch), epoch_train_loss, 'b8:', label='train_loss')
    ax.plot(np.arange(epoch), epoch_test_loss, 'Pg-', label='test_loss')
    ax.legend()
    plt.savefig('lossWEncoder.png')

    fig, ax = plt.subplots()
    ax.plot(np.arange(epoch), epoch_train_acc, 'b8:', label='train_acc')
    ax.plot(np.arange(epoch), epoch_test_acc, 'Pg-', label='test_acc')
    ax.legend()
    plt.savefig('accuracyWEncoder.png')


if __name__ == '__main__':
    main()