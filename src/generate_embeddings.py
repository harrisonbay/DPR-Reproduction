import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np
from collections import OrderedDict
import glob
import os


from model import *
from utils.wiki_data_loader import *
from utils.qa_pair_data_loader import *
from utils.evaluation_utils import *
from utils.utils import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--b", help="batch_size", default="32")
    parser.add_argument("--wiki", help="path to wikipedia DB")
    parser.add_argument("--qa_pair", help="path to qa pair csv")
    parser.add_argument("--world_size", help="world size")
    parser.add_argument("--v", help="experiment version")
    parser.add_argument("--model", help="DISTILBERT or BERT", default="BERT")
    args = parser.parse_args()

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '1234'

    if os.path.exists('./embeddings/'):
        pass
    else:
        os.makedirs('./embeddings/')


    print(torch.cuda.is_available())

    assert int(args.b) % int(args.world_size) == 0, "batch size must be divisible by world size"
    assert args.model == "DISTILBERT" or args.model == "BERT"

    mp.spawn(create_embeddings, nprocs=int(args.world_size), args=(args,))

def create_embeddings(gpu, args):

    args.world_size = int(args.world_size)
    torch.manual_seed(0)
    rank = gpu
    torch.cuda.set_device(rank)
    print(rank)

    log_interval = 100

    dist.init_process_group(backend='nccl',
                            init_method='env://',
                            world_size=args.world_size, rank=rank)

    # create wikipedia loader (it returns tokenized passages)
    wiki_set = WikiDataset(args.wiki)
    
    # refer below for explanation of drop_last = True
    wiki_sampler = torch.utils.data.distributed.DistributedSampler(wiki_set, 
                                                                   num_replicas=args.world_size,
                                                                   rank=rank, shuffle=False, 
                                                                   drop_last=True)
    wiki_loader = torch.utils.data.DataLoader(wiki_set,
                                              batch_size=int(args.b)//int(args.world_size),
                                              pin_memory=True,
                                              sampler=wiki_sampler, shuffle=False)

    # create questions loader (it returns tokenized questions)
    qa_pair_set = QAPairDataset(args.qa_pair)
    
    # Note drop_last=True. This is *absolutely necessary* for our code to
    # function properly: we do some hacky stuff related to question/psg indices
    # later in `evaluate.py` and `evaluation_utils.py`. If our world size is 4,
    # we're cutting off at maximum the last 3 questions in our dev and test
    # sets when we evaluate (if you don't add this, then the DistributedSampler
    # acts like a loop of data, and if your data set length is not divisble by
    # the world size, the last batches will loop around and give you extra
    # samples from the beginning. EX: world size 4, batch size = 1, n = 14: the
    # last iteration will distribute out questions 13, 14, 1, and 2 [1-indexed])
    qa_pair_sampler = torch.utils.data.distributed.DistributedSampler(qa_pair_set,
                                                                      num_replicas=args.world_size,
                                                                      rank=rank, shuffle=False,
                                                                      drop_last=True)
    qa_pair_loader = torch.utils.data.DataLoader(qa_pair_set,
                                                 batch_size=int(args.b)//int(args.world_size),
                                                 pin_memory=True,
                                                 sampler=qa_pair_sampler, shuffle=False)

    # concatenate the indices:
    # index 0: *global* index of the passage in the wikipedia database
    # index 1-768: the BERT embedding of the passage's CLS token
    psg_embeddings = np.zeros((0, 769), dtype=np.float32)

    # We do the same thing for the questions
    ques_embeddings = np.zeros((0, 769), dtype=np.float32)

    net = None
    if args.model == "DISTILBERT":
        net = DISTILBERT_QA().cuda(gpu)
    else:
        net = BERT_QA().cuda(gpu)

    checkpoints = sorted(glob.glob('./logs/' + args.v  + '/' + '/*.pt'), key=os.path.getmtime)

    # https://stackoverflow.com/a/44319982
    state_dict = torch.load(checkpoints[-1])
    new_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k[7:]
        new_dict[new_key] = v
    net.load_state_dict(new_dict)

    net = net.cuda(gpu)
    model = nn.parallel.DistributedDataParallel(net,
                                                device_ids=[gpu],
                                                find_unused_parameters=True)
    model.eval()

    print("==========embedding the passages==========")
    for batch_idx, (passage, psg_indices) in enumerate(wiki_loader):
        with torch.no_grad():
            passage = passage.long().cuda(non_blocking=True)
            _, p_emb = model(None, passage)
            
            np_psg_indices = np.expand_dims(psg_indices.numpy(), axis=1)
            # print(np_psg_indices)

            p_emb = p_emb.detach().cpu().numpy()
            p_emb = np.concatenate((np_psg_indices, p_emb), axis=1)
            # print(type(p_emb[0][0]))

            psg_embeddings = np.concatenate((psg_embeddings, p_emb), axis=0)

            if batch_idx % log_interval == 0:
                print(f'Embedded {batch_idx} batches of passages')

    print("==========embedding the questions==========")
    for batch_idx, (ques, ques_indices) in enumerate(qa_pair_loader):
        with torch.no_grad():
            ques = ques.long().cuda(non_blocking=True)

            q_emb, _ = model(ques, None)

            np_ques_indices = np.expand_dims(ques_indices.numpy(), axis=1)

            q_emb = q_emb.detach().cpu().numpy()
            q_emb = np.concatenate((np_ques_indices, q_emb), axis=1)
            ques_embeddings = np.concatenate((ques_embeddings, q_emb), axis=0)

            if batch_idx % log_interval == 0:
                print(f'Embedded {batch_idx} batches of questions')

    serialize_vectors(f"./embeddings/{args.v}-psg-{rank}.h5", psg_embeddings)
    serialize_vectors(f"./embeddings/{args.v}-ques-{rank}.h5", ques_embeddings)

if __name__ == '__main__':
    main()
