#!/x0/arnavmd/python3/bin/python3
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import argparse

from utils import *
from data_loader import *
from model import *

print(torch.__version__)
torch.cuda.set_device(1)



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--b", help="batch size", default="2")
    parser.add_argument("--e", help="number of epochs", default="100")
    parser.add_argument("--lr", help="initial learning rate", default="1e-5")
    parser.add_argument("--v", help="experiment version", default="0.1")
    parser.add_argument("--train_set", help="path to training dataset (h5 file)", default='/x0/arnavmd/nlp_proj/DPR/data/data/retriever/nq-train.json')
    parser.add_argument("--dev_set", help="path to training dataset (h5 file)", default='/x0/arnavmd/nlp_proj/DPR/data/data/retriever/nq-dev.json')
    parser.add_argument("--m", help="additional comments", default="")

    args = parser.parse_args()

    LEARNING_RATE = float(args.lr)
    EXPERIMENT_VERSION = args.v
    LOG_PATH = './logs/' + EXPERIMENT_VERSION + '/'


    torch.manual_seed(0)    
    print(torch.cuda.is_available())

    train_set = NQDataset(args.train_set)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=2, shuffle=True)

    dev_set = NQDataset(args.dev_set)
    dev_loader = torch.utils.data.DataLoader(dev_set, batch_size=2, shuffle=True)

    model = BERT_QA().cuda()
    print("Downloaded models")

   
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


    log_interval = 1
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0)
    min_loss = 10
    train_log = os.path.join(LOG_PATH, 'train_log.txt')
    test_log = os.path.join(LOG_PATH, 'test_log.txt')

    for epoch in range(int(args.e)):

        print("="*10 + "Epoch " + str(epoch) + "="*10)

        losses = []
        model.train()
        for batch_idx, (ques, pos_ctx, neg_ctx) in enumerate(train_loader):
            

            # TODO: clean this up (alternating positive/negative contexts)
            ques = ques.long().cuda()

            psg = torch.cat((pos_ctx, neg_ctx), dim=1)
            psg = psg.reshape((-1, 256)) 
            psg = psg.long().cuda()

            optimizer.zero_grad()
            q_emb, p_emb = model(ques, psg)
            loss = model.loss_fn(q_emb, p_emb)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:


                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), loss.item()))

                with open(train_log, 'a+') as f:
                    f.write(str(loss.item()) + "\n")


        test_losses = []
        model.eval()        
        for batch_idx, (ques, pos_ctx, neg_ctx) in enumerate(dev_loader):


            with torch.no_grad():
                ques = ques.long().cuda()

                psg = torch.cat((pos_ctx, neg_ctx), dim=1)
                psg = psg.reshape((-1, 512)) 
                psg = psg.long().cuda()

                q_emb, p_emb = model(ques, psg)
                loss = model.loss_fn(q_emb, p_emb)
                test_losses.append(loss.item())

        with open(test_log, 'a+') as f:
            f.write(str(np.mean(test_loss))) 

        print('Val Loss:', np.mean(test_loss))


        
if __name__ == '__main__':
    main()
