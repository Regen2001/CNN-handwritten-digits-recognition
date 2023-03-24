from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import os
import numpy as np
import shutil
from tqdm import trange
import time


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, 1)
        self.conv2 = nn.Conv2d(8, 16, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(2304, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

def main(args):

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if (torch.cuda.is_available() and args.use_cuda) else "cpu")

    train_loader, test_loader = dataset(args)

    model = Net().to(device)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    File = 'LOSS' + str(args.batch_size)
    fileCreate(File)

    accuracy = np.zeros([args.epochs, 1])

    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        accuracy[epoch-1] = test(model, device, test_loader)
        scheduler.step()
    end = time.perf_counter()
    # The time used to train is 109s, batch is 512
    # The time used to train is 169s, batch is 64

    print('The time used to train is ', round(end-start))

    # saveData('accuracy.npy', accuracy, 'accuracy')

    if args.save_model is True:
        fileCreate('best_model')
        torch.save(model.state_dict(), "best_model/mnist_cnn.pt")

    return 0

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

    parser.add_argument('--batch-size', type=int, default=512, metavar='N', help='input batch size for training (default: 512)')

    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')

    parser.add_argument('--epochs', type=int, default=15, metavar='N', help='number of epochs to train (default: 15)')

    parser.add_argument('--num-workers', type=int, default=6, metavar='N', help='number of worker of torch to train (default: 10)')

    parser.add_argument('--lr', type=float, default=1, metavar='LR', help='learning rate (default: 1.0)')

    parser.add_argument('--gamma', type=float, default=0.7, metavar='M', help='Learning rate step gamma (default: 0.7)')

    parser.add_argument('--use-cuda', action='store_true', default=True, help='disables CUDA training')

    parser.add_argument('--dry-run', action='store_true', default=False, help='quickly check a single pass')

    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=True, help='For Saving the current Model')

    return parser.parse_args()

def dataset(args):

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if torch.cuda.is_available() and args.use_cuda:
        cuda_kwargs = {'num_workers': args.num_workers,
                        'pin_memory': True,
                        'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Make train dataset split
    trainset = datasets.MNIST('./data', train=True, download=True,
                       transform=transform)
    # Make test dataset split
    testset = datasets.MNIST('./data', train=False,
                       transform=transform)

    train_loader = torch.utils.data.DataLoader(trainset,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(testset, **test_kwargs)

    return train_loader, test_loader

def train(args, model, device, train_loader, optimizer, epoch):

    model.train()
    sum_up_batch_loss = 0
    # lossFile = 'LOSS' + str(args.batch_size) + '/loss_epoch' + str(epoch) + '.npy'
    lossFile = 'LOSS' + str(args.batch_size) + '/loss' + '.npy'

    with trange(len(train_loader)) as pbar:
        for batch_idx, ((data, target), i) in enumerate(zip(train_loader, pbar)):
            pbar.set_description(f"epoch{epoch}/{args.epochs}")
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            sum_up_batch_loss += loss.cpu().detach().numpy()
            average_loss = sum_up_batch_loss/(batch_idx+1)
            pbar.set_postfix({'loss':'{:.4f}'.format(loss.cpu().detach().numpy()), 'average loss':'{:.4f}'.format(average_loss)})

            saveData(lossFile, loss.cpu().detach().numpy(), 'loss')

    return 0

def saveData(file, data, item_name):
    if os.path.exists(file) is True:
        dictionary = np.load(file, allow_pickle= True).item()
        data_temp = dictionary[item_name]
        data = np.append(data_temp, data)

    dictionary = {item_name: data}
    np.save(file, dictionary)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct_num = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            predict = output.argmax(dim=1, keepdim=True)
            correct_num += predict.eq(target.view_as(predict)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct_num / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(test_loss, correct_num, len(test_loader.dataset), 100.*accuracy))

    return accuracy

def fileCreate(fileName):
    if os.path.exists(fileName) is True:
        shutil.rmtree(fileName)
        os.makedirs(fileName)
    else:
        os.makedirs(fileName)


if __name__ == '__main__':
    args = parse_args()
    main(args)