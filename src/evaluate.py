import torch.utils.data
import torch
import os
import argparse

from utils.evaluation_utils import *
from utils.wiki_data_loader import *
from utils.qa_pair_data_loader import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wiki", help="path to wikipedia DB")
    parser.add_argument("--qa_pair", help="path to qa pair csv")
    parser.add_argument("--world_size", help="world size")
    parser.add_argument("--v", help="experiment version")
    args = parser.parse_args()

    args.world_size = int(args.world_size)

    if os.path.exists('./results/'):
        pass
    else:
        os.makedirs('./results/')
    
    passage_embeddings = []
    question_embeddings = []
    for i in range(args.world_size):
        passage_embeddings.append(deserialize_vectors(f"./embeddings/{args.v}-psg-{i}.h5").astype(np.float32))
        question_embeddings.append(deserialize_vectors(f"./embeddings/{args.v}-ques-{i}.h5").astype(np.float32))
    
    wiki_dataset = WikiDataset(args.wiki)
    qa_pair_dataset = QAPairDataset(args.qa_pair)

    ks = [1, 5, 20, 100, 500, 2000, 10000]
    results = evaluate_wiki(question_embeddings,
                            passage_embeddings,
                            wiki_dataset, qa_pair_dataset, ks=ks)
    
    file = open(f"./results/results-{args.v}.txt", "a")
    file.write(f"Experiment Version | Top-K | Accuracy\n")

    for i in range(len(ks)):
        # experiment version | top-k | accuracy
        file.write(f"{args.v}\t{ks[i]}\t{results[i]}\n")
    
    file.close()
if __name__ == '__main__':
    main()
