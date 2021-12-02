import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms

from models import VGG, DenseNet121, LeNet, ResNet18, ShuffleNetV2, \
            MobileNetV2, ResNet101, ResNet50, ResNet34, ResNet152, \
            DenseNet161, DenseNet169, DenseNet201, WideResNet40_10

from torch.nn.parallel import DistributedDataParallel as ddp
import torch.distributed as dist

import argparse
import time
import os

from get_comm_hooks import get_comm_hooks

from utils.pickle import save_obj, make_directory

def write_results(args, result):
    """
    Write recorded training metrics to files.
    :param args: Training args.
    :param res: Results of the training.
    """
    name = args.name
    directory = './results/' + name
    print('Writing results ({})..'.format(name))
    make_directory(directory)
    save_obj(result['train_losses'], directory + '/train_losses')
    save_obj(result['test_accuracies'], directory + '/test_accuracies')
    save_obj(result['time'], directory + '/time_per_epoch')

    if result["test_accuracies"]:
        best_acc = max(result["test_accuracies"])
    else:
        best_acc = 0

    my_args = dict(
        model=args.model,
        dataset=args.dataset,
        opt=args.opt,
        lr=args.lr,
        batch_size=args.bs,
        world_size=args.world_size,
        momentum=args.momentum,
        nesterov=args.nesterov,
        weight_decay=args.weight_decay,
        grad_reducer=args.grad_reducer,
        matrix_approximation_rank=args.matrix_approximation_rank,
        beta=args.beta,
        sketch_size=args.sketch_size,
        best_acc=best_acc
    )

    open_mode = 'w'
    if args.resume:
        open_mode = 'a'
    with open(directory + '/README.md', open_mode) as file:
        if args.resume:
            file.write('\n')
        for arg, val in my_args.items():
            file.write(str(arg) + ': ' + str(val) + '\\\n')


def get_data_loader(args, gpu):
    """
    Loads the required dataset
    :param dataset: Can be either 'cifar10' or 'cifar100'
    :param batch_size: The desired batch size
    :return: Tuple (train_loader, test_loader, num_classes)
    """
    # batch_size= batch_size // args.world_size
    dataset = args.dataset
    dataset = dataset.lower()
    batch_size = args.bs

    world_size = args.world_size
    rank = args.nr * args.gpus + gpu

    if dataset == "cifar10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    elif dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

    if dataset == "cifar10":
        num_classes = 10
        trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
    elif dataset == "cifar100":
        num_classes = 100
        trainset = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test)
    else:
        raise ValueError('Dataset not supported yet')

    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, num_replicas=world_size, rank=rank)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, sampler=train_sampler)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

    return trainloader, testloader



def get_model(args, gpu):
    """
    :param device: 'cuda' if you have a GPU, 'cpu' otherwise
    :param model_name: One of the models available in the folder 'models'
    :param num_classes: 10 or 100 depending on the chosen dataset
    :return: ddp wrapped model
    """

    model_name = args.model
    model_name = model_name.lower()
    dataset = args.dataset
    dataset = dataset.lower()

    if dataset == 'cifar100':
        num_classes = 100
    elif dataset == 'cifar10':
        num_classes = 10
    elif dataset == 'mnist':
        num_classes = 10
    else:
        raise ValueError('Dataset not supported yet')

    if args.nr * args.gpus + gpu == 0:
        print("network:" + model_name)

    if model_name == "vgg":
        net = VGG("VGG19", num_classes=num_classes)
    elif model_name == "lenet":
        net = LeNet()
    elif model_name == 'mobilenetv2':
        net = MobileNetV2()
    elif model_name == 'wideresnet':
        net = WideResNet40_10()
    elif model_name == "resnet18" or model_name == "resnet":
        net = ResNet18(num_classes=num_classes)
    elif model_name == "resnet50":
        net = ResNet50(num_classes=num_classes)
    elif model_name == "resnet34":
        net = ResNet34(num_classes=num_classes)
    elif model_name == "resnet101":
        net = ResNet101()
    elif model_name == "resnet152":
        net = ResNet152()
    elif model_name == "densenet121":
        net = DenseNet121()
    elif model_name == "densenet161":
        net = DenseNet161()
    elif model_name == "densenet169":
        net = DenseNet169()
    elif model_name == "densenet201":
        net = DenseNet201()
    elif model_name == 'shufflenetv2':
        net = ShuffleNetV2(1)
    else:
        raise ValueError("Model is currently not supported")

    device = torch.device("cuda:{}".format(gpu))
    net = net.to(device)
    ddp_net = ddp(net, device_ids=[gpu], output_device=gpu)

    return ddp_net


def get_optimizer(args, net, gpu):
    """
    Creates the right optimizer regarding to the parameters and attach it to the net's parameters.
    :param net: The net to optimize.
    :return: optimizer.
    """

    opt_name = args.opt
    opt_name = opt_name.lower()

    lr, momentum, weight_decay = args.lr, args.momentum, args.weight_decay
    nesterov = True if args.nesterov == 1 else False

    if args.nr * args.gpus + gpu == 0:
        print('Optimizer: ' + opt_name)

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)

    return optimizer


def train(net, trainloader, device, optimizer, criterion):
    """
    One epoch training of a network.
    :param net: The given network.
    :param trainloader: Pytorch DataLoader (train set)
    :param device: Either 'cuda' or 'cpu'
    :param optimizer: The used optimizer.
    :param criterion: The loss function.
    :return: train_loss
    """

    net.train()
    train_loss = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        n_batch = batch_idx

    loss = train_loss / (n_batch + 1)

    return loss


def test(net, testloader, device, optimizer, criterion):

    net.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

    acc = 100. * correct / total

    return acc


def construct_and_train(gpu, args):

    # parameters for distributed learning
    device = torch.device("cuda:{}".format(gpu))
    rank = args.nr * args.gpus + gpu
    print(rank)

    dist.init_process_group(backend='nccl', init_method=args.dist_url, world_size=args.world_size, rank=rank)

    torch.manual_seed(args.seed)

    # data preparation
    trainloader, testloader = get_data_loader(args, gpu)
    # get the model that is wrapped with ddp
    net = get_model(args, gpu)
    # register communication hooks for ddp
    comm_state = get_comm_hooks(args, net, gpu)
    optimizer = get_optimizer(args, net, gpu)
    criterion = nn.CrossEntropyLoss()

    result = dict(
        train_losses=[],
        test_accuracies=[],
        time=[]
    )

    try:
        for epoch in range(args.epochs):
            if epoch == args.lr_decay_at_epoch1 or epoch == args.lr_decay_at_epoch2 or epoch == args.lr_decay_at_epoch3:
                # reset the error feedback when decreasing step size
                # if 'ef' in args.grad_reducer.lower(): # conef or ef
                #     comm_state.clean_error()

                # decrease learning rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] /= args.lr_decay_ratio

            start_time = time.time()
            train_loss = train(net, trainloader, device, optimizer, criterion)
            epoch_time = time.time() - start_time
            if rank == 0:
                print('\nEpoch: %d' % epoch)
                test_acc = test(net, testloader, device, optimizer, criterion)
                print('test acc: %f' % test_acc)
                result["train_losses"].append(train_loss)
                result["test_accuracies"].append(test_acc)
                result['time'].append(epoch_time)

    except KeyboardInterrupt:
        print('Interrupting..')
    finally:
        if rank == 0:
            write_results(args, result)

    return result


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # 4 nodes setup
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='number of nodes')
    parser.add_argument('-g', '--gpus', default=4, type=int, help='number of gpus per node')
    parser.add_argument('--dist_url', default='tcp://10.0.207.146:25191', type=str, help='url for distributed training')
    parser.add_argument('-nr', '--nr', default=0, type=int, help='rank of a node')

    # model, dataset, checkpoints
    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset')

    parser.add_argument('--model', default='wideresnet', type=str, help='Model architecture')
    # choosing model as resnet18 or wideresnet or any neural networks you like
    parser.add_argument('-r', '--resume', default=0, type=int, help='resume from checkpoint')

    # general training setup
    parser.add_argument('--epochs', default=150, type=int, help='number of epochs')
    parser.add_argument('--bs', default=16, type=int, help='batch size')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')


    # optimizer setup: SGD is enough. EF or ConEF is coped separately in ddp
    parser.add_argument('--name', default='image_classification', type=str, help='checkpoint name')
    # currently only sgd is supported, but it is not hard to change to other optimizers
    parser.add_argument('--opt', default='SGD', type=str, help='Optimizer')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--lr_decay_ratio', default=10, type=int, help='decay lr ratio')
    parser.add_argument('--lr_decay_at_epoch1', default=80, type=int, help='decay lr once at this epoch')
    parser.add_argument('--lr_decay_at_epoch2', default=120, type=int, help='decay lr twice at this epoch')
    parser.add_argument('--lr_decay_at_epoch3', default=150, type=int, help='decay lr three times at this epoch')

    # resnet18 setup
    # parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    # parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
    # parser.add_argument('--nesterov', default=0, type=int, help='Using nesterov momentum or not')
    # wideresenet setup
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--nesterov', default=1, type=int, help='Using nesterov momentum or not')

    ## reducers and gradient compression setup. Supported choices are listed below explicitly.
    # ddp specific setups
    parser.add_argument('--start_compression_iter', default=10, type=int, help='iteration to start gradient compression in ddp')

    # SGD
    # parser.add_argument('--grad_reducer', default='allreduce', type=str, help='gradient compressor')
    # EFSGD
    # parser.add_argument('--grad_reducer', default='ef_powerSGD', type=str, help='gradient compressor')
    # parser.add_argument('--grad_reducer', default='ef_randomblock', type=str, help='gradient compressor')
    # ConEF
    parser.add_argument('--grad_reducer', default='conef_randomblock', type=str, help='gradient compressor')
    # parser.add_argument('--grad_reducer', default='conef_powerSGD', type=str, help='gradient compressor')
    parser.add_argument('--sketch_size', default=0.2, type=float, help='sketch_size for ConEF')
    parser.add_argument('--beta', default=0.9, type=float, help='beta for ConEF')
    # unbiased gradient compressors
    # parser.add_argument('--grad_reducer', default='unbiased_randomblock', type=str, help='gradient compressor')

    # powerSGD setups
    parser.add_argument('--matrix_approximation_rank', default=4, type=int, help='Matrix Rank in PowerSGD')
    parser.add_argument('--min_compression_rate', default=2, type=int, help='decide when a tensor is worth compressed')
    parser.add_argument('--orthogonalization_epsilon', default=0, type=int, help='decide when a tensor is worth compressed')

    # random-block-k setups
    parser.add_argument('--compression_ratio', default=0.1, type=float, help='compression ratio')


    args = parser.parse_args()

    args.world_size = args.gpus * args.nodes
    mp.spawn(construct_and_train, nprocs=args.world_size, args=(args,))
