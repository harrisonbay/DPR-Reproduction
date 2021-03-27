# Overview

This is a project that attempts to reproduce and verify the main claims of ["Dense Passage Retrieval for Open-Domain Question Answering"](https://arxiv.org/pdf/2004.04906.pdf) (Karpukhin et al., EMNLP 2020) with the final report following the format of submissions to the [ML Reproducibility Challenge](https://paperswithcode.com/rc2020). This project was done as a final project for [CSE 517 wi21](https://docs.google.com/document/d/1gBz2w79DBrGjNGq2TMqJBDIWzUGsQacWFAszZKz6OKI/edit), taught by Prof. Noah Smith.

# Dependencies

All of our dependencies are listed within the `environment.yml` file and it should be usable out-of-the-box if you install the environment with `conda`. However, if you do not wish to use the environment file, the main dependencies of our code are [Huggingface Transformers](https://huggingface.co/transformers/), [PyTorch](https://pytorch.org/), [pandas](https://pandas.pydata.org/), [h5py](https://www.h5py.org/), [FAISS (CPU version is fine)](https://github.com/facebookresearch/faiss), [seaborn](https://seaborn.pydata.org/), [matplotlib](https://matplotlib.org/), and optionally [Weights and Biases](https://wandb.ai/site).

# Installation and Setup

~~These are install instructions to get everything working on a fresh VM. The VM we used ran Debian GNU/Linux 10 (buster) as its OS, and in particular, we used an A2 machine with 4 Tesla A100s on Google Cloud Platform to reproduce selected results of the paper. As a warning, this paper's results (and subsequently the scripts included in this repository) *are expensive to run and reproduce*; we used roughly $___ worth of Google Cloud Platform credits to reproduce *selected, not all*, results in the original paper.~~ 

##### **Warning**: These installation instructions were not tested. We ultimately were not allowed to rent out a powerful-enough GPU-enabled machine on either Azure or GCP. The content is correct; the directories may not be.

### Install git

1. `sudo apt install git-all`
	* Answer "yes" if you're prompted.

2. `sudo source .bashrc`
	* This reloads your shell, and the git command should now work. We use this multiple times throughout the setup process.

### Experiment code

This should be cloned in the home directory of your VM.

1. `git clone https://github.com/peminguyen/CSE517-final-project.git`
	* We need environment.yml for setting up conda.

Note: from here on out in installation, there is a bash script that runs everything for you in the repository, `init.sh`. Simply move it out to your home directory with `mv CSE517-final-project/init.sh .`.

2. `cd CSE517-final-project`

3. `mkdir bert-base-uncased; cd bert-base-uncased`
 
4. `wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz`

5. `tar -xzvf bert-base-uncased.tar.gz`

### Setting up conda

Starting from your home directory,

1. `wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh`

2. `bash Miniconda3-latest-Linux-x86_64.sh`
	* Go through the prompts. Leaving everything as defaults should be fine. You should run conda init.

3. `source .bashrc`

4. `conda env create -f CSE517-final-project/environment.yml`
	* There's probably quite a few extraneous packages in this `environment.yml` file, but this is guaranteed to work. Feel free to prune stuff that you find isn't needed, though it'll probably be really time-consuming.

5. `conda activate DPR`

### Getting the data

Start from your home directory.

#### Preprocessed Natural Questions (NQ) data

1. `git clone https://github.com/facebookresearch/DPR.git`

2. `cd DPR/dpr`

3. `python3 data/download_data.py --resource data.retriever.qas --output_dir ~`

4. `python3 data/download_data.py --resource data.retriever.nq --output_dir ~`

#### EfficientQA subset of Wikipedia

1. `python3 data/download_data.py --resource data.wikipedia_split --output_dir ~`

2. `cd ~`

3. `git clone https://github.com/efficientqa/retrieval-based-baselines.git`

4. `cd retrieval-based-baselines; python3 filter_subset_wiki.py --db_path ~/downloads/data/wikipedia_split/psgs_w100.tsv --data_path ~/downloads/data/retriever/nq-train.json`



# Running the Experiments

Run the shell script `training.sh`. **Warning**: This shell script was not tested. We ultimately were not allowed to rent out a powerful-enough GPU-enabled machine on either Azure or GCP, so we never fully ran this shell script. The content is correct; the directories may not be. It's also probably a little bit outdated.

Note: There are some commented out portions of code in `training.py` that you can hook up to [Weights and Biases](https://wandb.ai/site).

If you're not using the shell script, once you have the data, the order of the experiment should be:

1. `training.py`: `python training.py --b <batch size, we use 32> --e <epochs, we use 40> --v <experiment version> --train_set <path to train set json> - dev_set <path to dev set json> --world_size <number of GPUs> --model <"DISTILBERT" or "BERT">` Note that you need to have bert-base-uncased downloaded and saved in a folder that is in the same directory as this file (see above for `wget` command).
2. `generate_embeddings.py`: `python generate_embeddings.py --b <batch size, we use 600> --wiki <path to wikipedia tsv> --qa_pair <path to question/answer pair csv> --world_size <number of GPUs> --v <experiment version> --model <"DISTILBERT" or "BERT">`
4. `evaluate.py`: `python evaluate.py --wiki <path to wikipedia tsv> --qa_pair <path to question/answer pair csv> --world_size <number of GPUs> --v <experiment version>`
5. `plot.py`: `python plot.py --v <experiment version>`

Note that there is no data pre-processing command.

# Experimental Results

The below statistics are for a DPR model trained for 40 epochs with batch size 32, evaluated on the Wikipedia subset:

|| Top-k = 20 | Top-k = 100 |
|-----| --- | ----------- |
|Dev.| 58.86% | 69.67% |
|Test| 60.48% | 71.09% |

We provide both the pretrained DISTILBERT and BERT models that were used to generate all presented results [here](https://drive.google.com/drive/folders/11MP71YhqX3mFXfXWB76jNR1DBjOC28tU?usp=sharing).
 

### Useful links/credits

* For getting a distributed version of our model up and running, we followed [this tutorial](https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html). There are some extra things that we needed to pay extra attention to--DDP is very finicky and the paper is very particular about how training should be set up (which is probably why they do not use any of the built-in PyTorch classes and instead implement their own DataIterators, etc.).

* The [DPR repository](https://github.com/facebookresearch/DPR) and the [retrieval-based baselines repository](https://github.com/efficientqa/retrieval-based-baselines).


