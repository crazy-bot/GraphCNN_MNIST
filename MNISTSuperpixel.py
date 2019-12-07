import os.path as osp
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import MNISTSuperpixels
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.utils import normalized_cut
from torch_geometric.nn import (NNConv, graclus, max_pool, max_pool_x,
                                global_mean_pool)
import matplotlib.pyplot as plt
import numpy as np

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'MNISTSuperpixel')
transform = T.Cartesian(cat=True)
train_dataset = MNISTSuperpixels(path, True, transform=transform)
test_dataset = MNISTSuperpixels(path, False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
d = train_dataset


def normalized_cut_2d(edge_index, pos):
    row, col = edge_index
    edge_attr = torch.norm(pos[row] - pos[col], p=2, dim=1)
    return normalized_cut(edge_index, edge_attr, num_nodes=pos.size(0))


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
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train(args,epoch):
    model.train()

    if epoch == 16:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001

    if epoch == 26:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001
    train_loss = 0
    correct = 0

    for batch_idx,data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(data.y).sum().item()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                epoch, batch_idx* len(data.y), len(train_loader.dataset), loss.item()))
            # print('Train Epoch: {} [batch_idx:{} datalen: {}/{}]\tLoss: {:.6f}'.format(
            #     epoch, batch_idx * len(data.y), len(train_loader.dataset), loss.item()))

    train_loss = train_loss / (batch_idx + 1)
    torch.save(model.state_dict(), 'model_superpixel.pt')
    print(
        '\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            train_loss, correct, len(train_loader.dataset),
            100. * correct / len(train_loader.dataset)))
    return train_loss, correct


def test():
    model.eval()
    correct = 0
    test_loss = 0

    # for data in test_loader:
    #     data = data.to(device)
    #     pred = model(data).max(1)[1]
    #     correct += pred.eq(data.y).sum().item()
    # return correct / len(test_dataset)

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, data.y,reduction='sum').item()
            pred = output.argmax(dim=1)
            correct += pred.eq(data.y).sum().item()

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
    use_cuda = True
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

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
    plt.savefig('loss.png')

    fig, ax = plt.subplots()
    ax.plot(np.arange(epoch), epoch_train_acc, 'b8:', label='train_acc')
    ax.plot(np.arange(epoch), epoch_test_acc, 'Pg-', label='test_acc')
    ax.legend()
    plt.savefig('accuracy.png')


if __name__ == '__main__':
    main()