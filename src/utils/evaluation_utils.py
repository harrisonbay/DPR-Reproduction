import os
import h5py
import numpy as np
import pandas as pd
import faiss
import regex as re
import string
import random

# from Facebook/DPR repo (Karpukhin et al, EMNLP 2020)
def _normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

# matrix should be size n x d
def serialize_vectors(filename, matrix):
    h5f = h5py.File(filename, 'w')
    h5f.create_dataset("data", data=matrix)
    h5f.close()

# reads in h5 of numpy array and returns it
def deserialize_vectors(filename):
    h5f = h5py.File(filename, 'r')
    out = h5f["data"][:]
    return out

# question_embeddings should be a list of matrices each of ~num_questions/world_size x d
# passage_embeddings should be a list of matrices each of ~num_psgs/world_size x d
def evaluate_wiki(question_embeddings, passage_embeddings, wiki_dataset, qa_pair_dataset, ks=[20, 100]):
    global_passage_embeddings = np.zeros((0, 769), dtype=np.float32)
    global_question_embeddings = np.zeros((0, 769), dtype=np.float32)
   
    for passage_embedding in passage_embeddings:
        global_passage_embeddings = np.concatenate((global_passage_embeddings,
                                                    passage_embedding), axis=0)
          
    for question_embedding in question_embeddings:
        print(question_embedding.shape)
        global_question_embeddings = np.concatenate((global_question_embeddings,
                                                  question_embedding), axis=0)
    
    # sort the passages and questions by the global indices
    print("sorting global passage embeddings")
    global_passage_embeddings = \
            global_passage_embeddings[np.argsort(global_passage_embeddings[:, 0])]
    print("sorting global question embeddings")
    global_question_embeddings = \
            global_question_embeddings[np.argsort(global_question_embeddings[:, 0])]

    # slice off the global indices because they aren't actually part of the
    # embedding
    global_passage_embeddings = np.ascontiguousarray(global_passage_embeddings[:, 1:])
    global_question_embeddings = np.ascontiguousarray(global_question_embeddings[:, 1:])
    print(global_passage_embeddings.shape)
    print(global_question_embeddings.shape)

    index = faiss.IndexFlatIP(global_passage_embeddings.shape[1])
    index.add(global_passage_embeddings)

    result_accs = []

    for k in ks:
        # results is num_questions x k
        _, results = index.search(global_question_embeddings, k)
        print(results.shape) 
        correct = 0
        for i in range(results.shape[0]):

            # WikiDataset.df['passage'][results[i, :]] will be a pandas series
            # object, so we convert to list
            psg_texts = wiki_dataset.df['passage'][results[i, :]].tolist()

            # these are the answers pertaining to the current question, indexed off
            # of the *global question index*
            answer_texts = qa_pair_dataset.df['answer'][i]
            
            # normalize the passages
            normalized_psgs = [_normalize_answer(psg) for psg in psg_texts]

            # normalize the answers. Make sure that pandas didn't do anything weird
            # with the strings in the answer list and cast them as strings
            normalized_answers = [_normalize_answer(str(answer)) for answer in answer_texts]

            # check to see if answer string in any of the passages
            if any([answer in passage for answer in normalized_answers for passage in normalized_psgs]):
                correct += 1

        print(f"top-{k} accuracy is {correct / results.shape[0]}")
        result_accs.append(correct / results.shape[0])
    return result_accs


# ====testing=====
# np.random.seed(1234)
# psgs = np.random.random((1000, 100)).astype('float32')
# questions = np.random.random((15, 100)).astype('float32')
# serialize_vectors(r"../test/psgs.h5", psgs);
# serialize_vectors(r"../test/questions.h5", questions);
# d_psgs = deserialize_vectors(r"../test/psgs.h5");
# d_questions = deserialize_vectors(r"../test/uestions.h5")
# print(np.array_equal(psgs, d_psgs))
# print(np.array_equal(questions, d_questions))
# evaluate_wiki([questions], [psgs])

