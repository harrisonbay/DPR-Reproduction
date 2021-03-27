#!/x0/arnavmd/python3/bin/python3
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import argparse
import torch.multiprocessing as mp
import torch.distributed as dist

from utils.utils import *
from utils.train_data_loader import *
from model import *
from utils.dist_utils import *
import wandb
# wandb.init()

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--b", help="batch size", default="32")
    parser.add_argument("--e", help="number of epochs", default="100")
    parser.add_argument("--lr", help="initial learning rate", default="1e-5")
    parser.add_argument("--v", help="experiment version", default="0.1")
    parser.add_argument("--train_set", help="path to training dataset (h5 file)", default='/x0/arnavmd/nlp_proj/DPR/data/data/retriever/nq-train.json')
    parser.add_argument("--dev_set", help="path to training dataset (h5 file)", default='/x0/arnavmd/nlp_proj/DPR/data/data/retriever/nq-dev.json')
    parser.add_argument("--m", help="additional comments", default="")
    parser.add_argument("--world_size", help="world size", default=4)
    parser.add_argument("--model", help="DISTILBERT or BERT", default="BERT")
    parser.add_argument("--top_k", help="for the hard negative sampling ablation", default=1)
    parser.add_argument("--seed", help="init seed", default=12345)
    # we added the default value of this to set_epoch() when we trained. I don't
    # think it matters from a correctness standpoint, but that might be why you
    # get slightly different results
    # parser.add_argument("--shuffle_seed", help="shuffle seed", default=1179493354)
    args = parser.parse_args()

    LEARNING_RATE = float(args.lr) * float(args.world_size)
    EXPERIMENT_VERSION = args.v
    LOG_PATH = './logs/' + EXPERIMENT_VERSION + '/'

    os.environ['MASTER_ADDR'] = '127.0.0.1' #'10.57.23.164'
    os.environ['MASTER_PORT'] = '1234'
    print(torch.cuda.is_available())

    assert int(args.b) % int(args.world_size) == 0, "batch size must be divisible by world size"
    assert args.model == "DISTILBERT" or args.model == "BERT"

    torch.manual_seed(int(args.seed))
    mp.spawn(train, nprocs=int(args.world_size), args=(args,))

def train(gpu, args):

    rank = gpu
    torch.cuda.set_device(rank)
    print(rank)
    dist.init_process_group(backend='nccl',
                            init_method='env://',
                            world_size=int(args.world_size),
                            rank=rank)


    train_set = NQDataset(args.train_set, k=int(args.top_k))
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set,
                                                                    num_replicas=int(args.world_size),
                                                                    rank=rank)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=int(args.b)//int(args.world_size),
                                               num_workers=0, pin_memory=True,
                                               sampler=train_sampler)

    dev_set = NQDataset(args.dev_set, k=int(args.top_k))
    dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_set,
                                                                  num_replicas=int(args.world_size),
                                                                  rank=rank)
    dev_loader = torch.utils.data.DataLoader(dev_set,
                                             batch_size=int(args.b)//int(args.world_size),
                                             num_workers=0, pin_memory=True,
                                             sampler=dev_sampler)
    net = None
    if args.model == "DISTILBERT":
        net = DISTILBERT_QA().cuda(gpu)
    else:
        net = BERT_QA().cuda(gpu)

    model = nn.parallel.DistributedDataParallel(net,
                                                device_ids=[gpu], 
                                                find_unused_parameters=True)
    print("Downloaded models")

    LOG_PATH = './logs/' + args.v  + '/'
    LEARNING_RATE = float(args.lr)

    if os.path.exists(LOG_PATH):
        restore_latest(model, LOG_PATH)
    else:
        os.makedirs(LOG_PATH)

    with open(os.path.join(LOG_PATH, 'setup.txt'), 'a+') as f:
        f.write("\nVersion: " + args.v)
        f.write("\nBatch Size: " + args.b)
        f.write("\nInitial Learning Rate: " + args.lr)
        f.write("\nTraining Set: " + args.train_set)
        f.write("\nValidation Set: " + args.dev_set)
        f.write("\nComments: " + args.m)

    log_interval = 20
    max_score = -1
    train_log = os.path.join(LOG_PATH, 'train_log.txt')
    dev_log = os.path.join(LOG_PATH, 'dev_log.txt')

    print(rank)

    warmup_counter = 0
    for epoch in range(int(args.e)):

        print("="*10 + "Epoch " + str(epoch) + "="*10)
        
        # Since our task is contrastive, we *need* batches to be random across
        # epochs because otherwise we have the same in-batch negatives for every
        # question; this makes our model generalize terribly. We assumed that
        # DistributedDataSampler would do this automatically for us, but we got
        # 33% top-100 accuracy on a 40 epoch batch size 32 model which is what
        # caused us to take a second look here. It turns out you *need* to
        # set epoch here in order to guarantee that batches are different from
        # epoch to epoch
        if int(args.world_size) > 1:
            train_sampler.set_epoch(epoch)
        losses = []
        model.train()
        for batch_idx, (ques, pos_ctx, neg_ctx) in enumerate(train_loader):

            warmup_counter += 1
            this_learning_rate = LEARNING_RATE
            # linear learning rate warmup over first 500 batches
            if warmup_counter < 500:
                this_learning_rate *= (warmup_counter / 500)

            optimizer = optim.Adam(model.parameters(), lr=this_learning_rate,
                                   weight_decay=0, eps=1e-8)

            ques = ques.long().cuda(non_blocking=True)

            # we concatenate our positive and negative contexts received
            # from our train_loader with the convention that the *positive*
            # context is on the even indices and the hard negative contexts
            # on the odd indices. We later use this convention as a nice way
            # to implement the in-batch negative sampling.
            psg = torch.cat((pos_ctx, neg_ctx), dim=1)
            psg = psg.reshape((-1, 256))
            psg = psg.long().cuda(non_blocking=True)

            optimizer.zero_grad()
            q_emb, p_emb = model(ques, psg)

            # calc_loss does all of the loss calculation for us; it 
            loss = calc_loss(rank, net, q_emb, p_emb)
            loss.mean().backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                # if rank == 0:
                #    wandb.log({'train_loss': loss.item()})

                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), loss.item()))

                with open(train_log, 'a+') as f:
                    f.write(str(loss.item()) + "\n")


        dev_losses = []
        dev_scores = []
        model.eval()
        for batch_idx, (ques, pos_ctx, neg_ctx) in enumerate(dev_loader):
            with torch.no_grad():
                ques = ques.long().cuda(non_blocking=True)

                psg = torch.cat((pos_ctx, neg_ctx), dim=1)
                psg = psg.reshape((-1, 256))
                psg = psg.long().cuda(non_blocking=True)

                q_emb, p_emb = model(ques, psg)
                sim, idx = net.get_sim(q_emb.detach(), p_emb.detach())


                preds = np.argmax(sim.detach().cpu().numpy(), axis=1)
                score = np.sum(preds == idx.cpu().detach().numpy())/len(preds)
                dev_scores.append(score)

                loss = net.loss_fn(sim, idx)
                dev_losses.append(loss.item())


        if rank == 0 and np.mean(dev_scores) > max_score:
            min_loss = np.mean(dev_scores)
            save(model, os.path.join(LOG_PATH, '%03d.pt' % epoch), num_to_keep=1)

        # if rank == 0:
        #    wandb.log({'val_loss': np.mean(dev_losses)})
        #    wandb.log({'val_acc': np.mean(dev_scores)})
        with open(dev_log, 'a+') as f:
            f.write(str(np.mean(dev_losses)))

        print('Val Loss:', np.mean(dev_losses), ' Val Acc:', np.mean(dev_scores))

    if rank == 0:
        save(model, os.path.join(LOG_PATH, '%03d.pt' % epoch), num_to_keep=2)

if __name__ == '__main__':
    main()
