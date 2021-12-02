# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx
import torch.optim as optim

import data
import model
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as ddp

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
    save_obj(result['test_ppl'], directory + '/test_ppl')
    save_obj(result['time'], directory + '/time')

    if result['test_ppl']:
        best_acc = min(result['test_ppl'])
    else:
        best_acc = 0

    my_args = dict(
        model=args.model,
        dataset=args.data,
        emsize=args.emsize,
        nhid=args.nhid,
        nlayers=args.nlayers,
        lr=args.lr,
        clip=args.clip,
        batch_size=args.batch_size,
        bptt=args.bptt,
        dropout=args.dropout,
        world_size=args.world_size,
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


def batchify(data, bsz, device):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


def get_data(corpus, device, args):
    eval_batch_size = 10
    global_batch_size = args.batch_size * args.world_size
    train_data = batchify(corpus.train, global_batch_size, device)
    val_data = batchify(corpus.valid, eval_batch_size, device)
    test_data = batchify(corpus.test, eval_batch_size, device)
    return train_data, val_data, test_data



def get_optimizer(args, net, gpu):
    """
    Creates the right optimizer regarding to the parameters and attach it to the net's parameters.
    :param net: The net to optimize.
    :return: A Pytorch optimizer.
    """

    opt_name = args.opt.lower()
    if args.nr * args.gpus + gpu == 0:
        print('Optimizer: ' + opt_name)

    lr,  weight_decay = args.lr, args.weight_decay

    if opt_name == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=weight_decay)
    elif opt_name == 'sgd':
        momentum = args.momentum
        nesterov = False if args.nesterov == 0 else True
        optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov)
    else:
        raise ValueError("Optimizer not supported yet.")

    return optimizer


def get_model(corpus, gpu, args):
    model_name = args.model.lower()
    ntokens = len(corpus.dictionary)
    if model_name == 'transformer':
        net = model.TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout)
    else:
        net = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)

    device = torch.device("cuda:{}".format(gpu))
    net = net.to(device)
    ddp_net = ddp(net, device_ids=[gpu], output_device=gpu)
    return ddp_net


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch_train(source, i, rank, args):
    # unit = args.batch_size // args.world_size
    unit = args.batch_size
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len , rank*unit : (rank+1)*unit]
    target_tmp = source[i+1:i+1+seq_len, rank*unit : (rank+1)*unit]
    target = torch.clone(target_tmp).detach().view(-1)
    return data, target


def get_batch_eval(source, i, args):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def evaluate(model, corpus, data_source, criterion, args):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    eval_batch_size = 10
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    if args.model != 'Transformer':
        hidden = model.module.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch_eval(data_source, i, args)
            if args.model == 'Transformer':
                output = model(data)
                output = output.view(-1, ntokens)
            else:
                output, hidden = model(data, hidden)
                hidden = repackage_hidden(hidden)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)


def train(model, optimizer, corpus, train_data, criterion, rank, args):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    if args.model != 'Transformer':
        hidden = model.module.init_hidden(args.batch_size)

    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch_train(train_data, i, rank, args)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        model.zero_grad()
        if args.model == 'Transformer':
            output = model(data)
            output = output.view(-1, ntokens)
        else:
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)
        loss = criterion(output, targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        total_loss += loss.item()


def main_func(gpu, args):
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    device = torch.device("cuda:{}".format(gpu))
    rank = args.nr * args.gpus + gpu
    print('rank ', rank)

    dist.init_process_group(backend='nccl', init_method=args.dist_url, world_size=args.world_size, rank=rank)

    ###############################################################################
    # Load data
    ###############################################################################

    result = dict(
        test_ppl=[],
        time=[]
    )

    corpus = data.Corpus(args.data)
    train_data, val_data, test_data = get_data(corpus, device, args)
    if rank == 0:
        print('Data loading is done')
    net = get_model(corpus, gpu, args)
    if rank == 0:
        print('Model preparation is done')
    optimizer = get_optimizer(args, net, gpu)
    comm_state = get_comm_hooks(args, net, gpu)
    if rank == 0:
        print('Optimizer is done')

    criterion = nn.CrossEntropyLoss()

    # Loop over epochs.
    lr = args.lr
    best_val_loss = None

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(0, args.epochs):
            if rank == 0:
                print('iteration started')
            if epoch == args.lr_decay_at_epoch1 or epoch == args.lr_decay_at_epoch2 or epoch == args.lr_decay_at_epoch3:
                for param_group in optimizer.param_groups:
                    param_group['lr'] /= args.lr_decay_ratio
                    lr /= args.lr_decay_ratio
                # if 'ef' in args.grad_reducer.lower():
                #     comm_state.clean_error()

            epoch_start_time = time.time()
            train(net, optimizer, corpus, train_data, criterion, rank, args)
            epoch_time = time.time() - epoch_start_time
            val_loss = evaluate(net, corpus, val_data, criterion, args)
            if rank == 0:
                result['time'].append(epoch_time)
                print('-' * 60)
                print('| Epoch {:3d} | lr: {} | time: {:5.2f}s | valid loss {:5.2f} | '
                        'valid ppl {:8.2f}'.format(epoch, lr, epoch_time, val_loss, math.exp(val_loss)))
                print('-' * 60)
            # Save the model if the validation loss is the best we've seen so far.
            # if not best_val_loss or val_loss < best_val_loss:
            #     # with open(args.save, 'wb') as f:
            #     #     torch.save(model, f)
            #     best_val_loss = val_loss
            # else:
            #     # Anneal the learning rate if no improvement has been seen in the validation dataset.
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] /= 4
            #         lr /= 4

            # Run on test data.
            if rank == 0:
                test_loss = evaluate(net, corpus, test_data, criterion, args)
                result['test_ppl'].append(math.exp(test_loss))
                print('=' * 60)
                print('| Testing epoch {:3d} | test loss {:5.2f} | test ppl {:8.2f}'.format(
                    epoch, test_loss, math.exp(test_loss)))
                print('=' * 60)
    except KeyboardInterrupt:
        print('-' * 70)
        print('Exiting from training early')
    finally:
        if rank == 0:
            write_results(args, result)

    # # Load the best saved model.
    # with open(args.save, 'rb') as f:
    #     model = torch.load(f)
    #     # after load the rnn params are not a continuous chunk of memory
    #     # this makes them a continuous chunk, and will speed up forward pass
    #     # Currently, only rnn model supports flatten_parameters function.
    #     if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
    #         model.rnn.flatten_parameters()



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model')

    # 4 nodes setup
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='number of nodes')
    parser.add_argument('-g', '--gpus', default=4, type=int, help='number of gpus per node')
    parser.add_argument('--dist_url', default='tcp://10.0.206.132:30418', type=str, help='url for distributed training')
    parser.add_argument('-nr', '--nr', default=0, type=int, help='rank of a node')

    parser.add_argument('--name', default='WordLanguageModel', type=str, help='checkpoint name')
    parser.add_argument('-r', '--resume', default=0, type=int, help='resume from checkpoint')
    parser.add_argument('--data', type=str, default='./data/wikitext-2', help='location of the data corpus')
    parser.add_argument('--model', type=str, default='LSTM', help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
    parser.add_argument('--emsize', type=int, default=672, help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=672, help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=2, help='number of layers')
    parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
    parser.add_argument('--bptt', type=int, default=35, help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--tied', action='store_true', help='tie the word embedding and softmax weights')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N', help='report interval')
    parser.add_argument('--save', type=str, default='model.pt', help='path to save the final model')
    parser.add_argument('--onnx-export', type=str, default='', help='path to export the final model in onnx format')
    parser.add_argument('--nhead', type=int, default=2, help='the number of heads in the encoder/decoder of the transformer model')
    # parser.add_argument('--dry-run', action='store_true', help='verify the code and the model')

    # optimizer
    parser.add_argument('--opt', default='SGD', type=str, help='Optimizer')
    parser.add_argument('--epochs', type=int, default=60, help='upper epoch limit')
    parser.add_argument('--lr', type=float, default=2, help='initial learning rate')
    parser.add_argument('--lr_decay_at_epoch1', default=15, type=int, help='decay lr 1st at epoch')
    parser.add_argument('--lr_decay_at_epoch2', default=30, type=int, help='decay lr 2nd at epoch')
    parser.add_argument('--lr_decay_at_epoch3', default=45, type=int, help='decay lr 3rd at epoch')
    parser.add_argument('--lr_decay_ratio', default=4, type=int, help='decay lr ratio')
    parser.add_argument('--batch_size', type=int, default=2, metavar='N', help='local batch size')
    # total batch size = local batch size * world size
    parser.add_argument('--weight_decay', default=0, type=float, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='heavy ball momentum')
    parser.add_argument('--nesterov', default=1, type=int, help='enbale nesterov momentum')

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

    # # unbiased gradient compressors
    # parser.add_argument('--grad_reducer', default='unbiased_randomblock', type=str, help='gradient compressor')
    # parser.add_argument('--grad_reducer', default='unbiased_mtrx', type=str, help='gradient compressor')
    # special
    # parser.add_argument('--grad_reducer', default='hEFSGD_powerSGD', type=str, help='gradient compressor')


    # powerSGD setups
    parser.add_argument('--matrix_approximation_rank', default=4, type=int, help='Matrix Rank in PowerSGD')
    parser.add_argument('--min_compression_rate', default=2, type=int, help='decide when a tensor is worth compressed')
    parser.add_argument('--orthogonalization_epsilon', default=0, type=int, help='decide when a tensor is worth compressed')

    # random-k setups
    parser.add_argument('--compression_ratio', default=0.1, type=float, help='compression ratio')

    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes

    mp.spawn(main_func, nprocs=args.gpus, args=(args,))