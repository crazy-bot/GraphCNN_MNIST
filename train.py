import pyximport; pyximport.install()
import argparse
import torch
from torchvision import datasets,transforms
import Network
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import Dataset as dset
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):

        #print(target.shape)
        coord,avg_values,A_spatial = dset.prepareGraph(data)
        #print(coord.shape,avg_values.shape,A_spatial.shape)
        data = [torch.from_numpy(np.concatenate((coord, avg_values), axis=2)).float().cuda(),
                torch.from_numpy(A_spatial).float().cuda(), False]
        #print(len(data))
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))


    train_loss = train_loss/(batch_idx+1)
    torch.save(model.state_dict(),'model.pt')
    print(
        '\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            train_loss, correct, len(train_loader.dataset),
            100. * correct / len(train_loader.dataset)))
    return train_loss,correct

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            coord, avg_values, A_spatial = dset.prepareGraph(data)
            data = [torch.from_numpy(np.concatenate((coord, avg_values), axis=2)).float().cuda(),
                    torch.from_numpy(A_spatial).float().cuda(), False]
            target = target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target,reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

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
    parser.add_argument('--epochs', type=int, default=100,
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

    MNIST_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True, transform= MNIST_transform),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader( datasets.MNIST('../data', train=False, transform= MNIST_transform),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

    model = Network.GNN()
    model.to(device)
    print(model)
    #model.load_state_dict(torch.load('model.pt'))
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-1 )
    print('number of trainable parameters: %d' %
          np.sum([np.prod(p.size()) if p.requires_grad else 0 for p in model.parameters()]))

    epoch_train_loss = []
    epoch_test_loss = []
    epoch_train_acc = []
    epoch_test_acc = []
    for epoch in range(1, args.epochs + 1):
        train_loss,train_correct = train(args, model, device, train_loader, optimizer, epoch)
        test_loss,test_correct = test(model, device, test_loader)
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
