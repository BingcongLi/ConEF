from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import Multi30k
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import argparse

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as ddp

from train_n_test import train_epoch, evaluate, get_text_transform
import time
from get_data import get_data
from utils.pickle import save_obj, make_directory
from get_comm_hooks import get_comm_hooks
from model import Seq2SeqTransformer



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
    save_obj(result['val_loss'], directory + '/val_loss')
    save_obj(result['time'], directory + '/time')
    save_obj(result['train_loss'], directory + '/train_loss')

    if result['val_loss']:
        best_loss = min(result['val_loss'])
    else:
        best_acc = 0

    my_args = dict(
        EMB_SIZE=args.EMB_SIZE,
        NHEAD=args.NHEAD,
        FFN_HID_DIM=args.FFN_HID_DIM,
        NUM_ENCODER_LAYERS=args.NUM_ENCODER_LAYERS,
        NUM_DECODER_LAYERS=args.NUM_DECODER_LAYERS,
        DROPOUT=args.DROPOUT,
        lr=args.lr,
        batch_size=args.BATCH_SIZE,
        world_size=args.world_size,
        grad_reducer=args.grad_reducer,
        matrix_approximation_rank=args.matrix_approximation_rank,
        beta=args.beta,
        sketch_size=args.sketch_size,
        best_loss=best_loss
    )

    open_mode = 'w'
    if args.resume:
        open_mode = 'a'
    with open(directory + '/README.md', open_mode) as file:
        if args.resume:
            file.write('\n')
        for arg, val in my_args.items():
            file.write(str(arg) + ': ' + str(val) + '\\\n')


def get_model(gpu, args):

    transformer = Seq2SeqTransformer(args.NUM_ENCODER_LAYERS, args.NUM_DECODER_LAYERS, args.EMB_SIZE,
                                     args.NHEAD, args.SRC_VOCAB_SIZE, args.TGT_VOCAB_SIZE, args.FFN_HID_DIM,
                                     args.DROPOUT)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    device = torch.device("cuda:{}".format(gpu))
    transformer = transformer.to(device)
    ddp_net = ddp(transformer, device_ids=[gpu], output_device=gpu)
    return ddp_net


def get_data(args):
    # Place-holders
    token_transform = {}
    vocab_transform = {}

    # Create source and target language tokenizer. Make sure to install the dependencies.
    # pip install -U spacy
    # python -m spacy download en_core_web_sm
    # python -m spacy download de_core_news_sm
    token_transform[args.SRC_LANGUAGE] = get_tokenizer('spacy', language='de_core_news_sm')
    token_transform[args.TGT_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')

    # helper function to yield list of tokens
    def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
        language_index = {args.SRC_LANGUAGE: 0, args.TGT_LANGUAGE: 1}

        for data_sample in data_iter:
            yield token_transform[language](data_sample[language_index[language]])

    # Define special symbols and indices
    UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
    # Make sure the tokens are in order of their indices to properly insert them in vocab
    special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

    for ln in [args.SRC_LANGUAGE, args.TGT_LANGUAGE]:
        # Training data Iterator
        train_iter = Multi30k(split='train', language_pair=(args.SRC_LANGUAGE, args.TGT_LANGUAGE))
        # Create torchtext's Vocab object
        vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                        min_freq=1,
                                                        specials=special_symbols,
                                                        special_first=True)

    # Set UNK_IDX as the default index. This index is returned when the token is not found.
    # If not set, it throws RuntimeError when the queried token is not found in the Vocabulary.
    for ln in [args.SRC_LANGUAGE, args.TGT_LANGUAGE]:
        vocab_transform[ln].set_default_index(UNK_IDX)

    return token_transform, vocab_transform


def main_func(gpu, args):

    result = dict(
        val_loss=[],
        time=[],
        train_loss=[]
    )

    UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3

    ######################################################################
    # Let's now define the parameters of our model and instantiate the same. Below, we also
    # define our loss function which is the cross-entropy loss and the optmizer used for training.

    torch.manual_seed(args.seed)
    DEVICE = torch.device("cuda:{}".format(gpu))
    rank = args.nr * args.gpus + gpu
    print('rank ', rank)

    dist.init_process_group(backend='nccl', init_method=args.dist_url, world_size=args.world_size, rank=rank)

    args.SRC_LANGUAGE = 'de'
    args.TGT_LANGUAGE = 'en'

    token_transform, vocab_transform = get_data(args)

    args.SRC_VOCAB_SIZE = len(vocab_transform[args.SRC_LANGUAGE])
    args.TGT_VOCAB_SIZE = len(vocab_transform[args.TGT_LANGUAGE])
    #################

    transformer = get_model(gpu, args)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    text_transform = get_text_transform(token_transform, vocab_transform)
    comm_state = get_comm_hooks(args, transformer, gpu)

    for epoch in range(args.epochs):
        if epoch == args.lr_decay_at_epoch1 or epoch == args.lr_decay_at_epoch2 or epoch == args.lr_decay_at_epoch3:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= args.lr_decay_ratio

        start_time = time.time()
        train_loss = train_epoch(text_transform, transformer, optimizer, loss_fn, args, gpu)
        per_epoch_time = time.time() - start_time
        if rank == 0:
            val_loss = evaluate(text_transform, transformer, loss_fn, gpu, args)
            print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {per_epoch_time:.3f}s"))
            result['time'].append(per_epoch_time)
            result['val_loss'].append(val_loss)
            result['train_loss'].append(train_loss)

    if rank == 0:
        write_results(args, result)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model')

    # 4 nodes setup
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='number of nodes')
    parser.add_argument('-g', '--gpus', default=1, type=int, help='number of gpus per node')
    parser.add_argument('--dist_url', default='tcp://10.0.206.132:20418', type=str, help='url for distributed training')

    parser.add_argument('--name', default='SGD', type=str, help='checkpoint name')
    parser.add_argument('-r', '--resume', default=0, type=int, help='resume from checkpoint')

    parser.add_argument('--EMB_SIZE', type=int, default=512, help='size of word embeddings')
    # parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    parser.add_argument('--NHEAD', type=int, default=8, help='multi-head attention')
    parser.add_argument('--FFN_HID_DIM', type=int, default=512, help='feed forward hidden dimension')
    parser.add_argument('--NUM_ENCODER_LAYERS', type=int, default=3, help='number of encoder layers')
    parser.add_argument('--NUM_DECODER_LAYERS', type=int, default=3, help='number of decoder layers')
    parser.add_argument('--DROPOUT', type=float, default=0.1, help='dropout applied to layers (0 = no dropout)')


    # optimizer
    parser.add_argument('--opt', default='adam', type=str, help='Optimizer')
    parser.add_argument('--epochs', type=int, default=30, help='upper epoch limit')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--lr_decay_at_epoch1', default=15, type=int, help='decay lr 1st at epoch')
    parser.add_argument('--lr_decay_at_epoch2', default=25, type=int, help='decay lr 2nd at epoch')
    parser.add_argument('--lr_decay_at_epoch3', default=30, type=int, help='decay lr 3rd at epoch')
    parser.add_argument('--lr_decay_ratio', default=4, type=int, help='decay lr ratio')
    parser.add_argument('--BATCH_SIZE', type=int, default=128, metavar='N', help='local batch size')

    ## reducers and gradient compression setup. Supported choices are listed below explicitly.
    # ddp specific setups
    parser.add_argument('--start_compression_iter', default=10, type=int,
                        help='iteration to start gradient compression in ddp')

    # ADAM
    # parser.add_argument('--grad_reducer', default='allreduce', type=str, help='gradient compressor')
    # EF-ADAM
    # parser.add_argument('--grad_reducer', default='ef_powerSGD', type=str, help='gradient compressor')
    # parser.add_argument('--grad_reducer', default='ef_randomblock', type=str, help='gradient compressor')
    # ConEF-ADAM
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