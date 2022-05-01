import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from model_cnn import *
from torchsummary import summary


def tmp_func(x):
    return x.repeat(3, 1, 1)


def normalize_data_mnist():
    """
    Get the normalize picture (MNIST and FMNIST)
    make the gray_picture * 3 layers
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Lambda(tmp_func),
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    return transform


def load_data(args):
    """
    Get the train and test dataloader
    """
    transform_mnist = normalize_data_mnist()

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform_mnist)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform_mnist)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    return trainloader, testloader, len(trainset), len(testset)


def init(args):
    """
    Make the net/device/data/criterion/optimizer
    """
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    trainloader, testloader, train_data_num, test_data_num = load_data(args)

    net = CNN().to(device)

    # Define loss functions and optimization
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
    # optimizer = optim.Adagrad(net.parameters(), lr=args.learning_rate, weight_decay=5e-4)
    # optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    optimizer = optim.RMSprop(net.parameters(), lr=args.learning_rate, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
    return device, trainloader, testloader, net, criterion, optimizer, train_data_num, test_data_num
