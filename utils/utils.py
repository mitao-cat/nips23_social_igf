import argparse
import os
import random
import torch
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='settings for running the implementation of ...')
    parser.add_argument('--seed', type=int, default=0, help='random seed setting')
    parser.add_argument('--model', type=str, default='bprmf', help='model')
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_epoch', type=int, default=50)
    parser.add_argument('--dataset', type=str, default='KuaiRec')
    parser.add_argument('--reg', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--numneg', type=int, default=1)
    # paretomtl
    parser.add_argument('--pretrain', type=int, default=1)
    parser.add_argument('--mode', type=str, default='sp')
    parser.add_argument('--npref', type=int, default=5)
    parser.add_argument('--thre', type=float, default=0.4)
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--pref_idx', type=int, default=0)
    parser.add_argument('--initsol_epoch', type=int, default=20)
    parser.add_argument('--initsol_lr', type=float, default=1e-3)
    parser.add_argument('--reg1', type=float, default=1)
    parser.add_argument('--reg2', type=float, default=1)
    return parser.parse_args()


def setup_seed(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def random_sampler(users, pos_items, train_list, args):
    batchsize, num_item = len(users), args.num_item
    neg_candidate_item = np.random.randint(num_item, size=batchsize*3)
    neg_items = np.zeros_like(pos_items)
    candidate_idx = 0
    for idx, u in enumerate(users):
        selected_neg = neg_candidate_item[candidate_idx]
        while selected_neg in train_list[u]:
            candidate_idx += 1
            selected_neg = neg_candidate_item[candidate_idx]
        neg_items[idx] = selected_neg
        candidate_idx += 1
    return torch.LongTensor(neg_items)
