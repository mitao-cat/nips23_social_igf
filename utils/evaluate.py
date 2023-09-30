import heapq
import numpy as np
from math import log


def f1(a,b):
    return 2*a*b/(a+b)


def rsd(x_list):
    return np.std(x_list) / np.mean(x_list)


def sigmoid(x_list):
    x_arr = np.array(x_list)
    return (1 + np.tanh(x_arr / 2)) / 2


def deg(vec):
    vec = np.array(vec) / np.linalg.norm(vec)
    return 180*np.arccos(vec[0])/np.pi


def NDCG_at_k(predicted_list, ground_truth, k):
    dcg_value = [(v / log(i + 1 + 1, 2)) for i, v in enumerate(predicted_list[:k])]
    dcg = np.sum(dcg_value)
    if len(ground_truth) < k:
        ground_truth += [0 for i in range(k - len(ground_truth))]
    idcg_value = [(v / log(i + 1 + 1, 2)) for i, v in enumerate(ground_truth[:k])]
    idcg = np.sum(idcg_value)
    return dcg / idcg


def user_precision_recall_ndcg(top50, test):
    dcg_list = []
    count_10, count_20, count_50 = 0, 0, 0
    for i in range(50):
        if i < 10 and top50[i] in test:
            count_10 += 1.0
        if i < 20 and top50[i] in test:
            count_20 += 1.0
        if top50[i] in test:
            count_50 += 1.0
            dcg_list.append(1)
        else:
            dcg_list.append(0)
    # calculate NDCG@k
    idcg_list = [1 for i in range(len(test))]
    ndcg_10 = NDCG_at_k(dcg_list, idcg_list, 10)
    ndcg_20 = NDCG_at_k(dcg_list, idcg_list, 20)
    ndcg_50 = NDCG_at_k(dcg_list, idcg_list, 50)
    precision_10 = count_10 / 10
    precision_20 = count_20 / 20
    precision_50 = count_50 / 50
    l = max(1,len(test))
    recall_10 = count_10 / l
    recall_20 = count_20 / l
    recall_50 = count_50 / l
    return np.array([precision_10, precision_20, precision_50]), \
           np.array([recall_10, recall_20, recall_50]), \
           np.array([ndcg_10, ndcg_20, ndcg_50])


def evaluate(Rec, val_dict, train_list):
    precision, recall, ndcg = np.zeros(3), np.zeros(3), np.zeros(3)
    result = np.zeros((3,3))
    for u in range(Rec.shape[0]):
        like_item = np.array(train_list[u])
        Rec[u, like_item] = -114514

    num_user = len(val_dict)
    for u in val_dict:  # iterate each user
        u_test = np.array(val_dict[u])
        heap = list(Rec[u])
        top50 = heapq.nlargest(50, range(len(heap)), heap.__getitem__) 
        if len(u_test) != 0:
            precision_u, recall_u, ndcg_u = user_precision_recall_ndcg(top50, u_test)
            precision += precision_u
            recall += recall_u
            ndcg += ndcg_u
        else:
            num_user -= 1
    precision /= num_user
    recall /= num_user
    ndcg /= num_user
    result[0], result[1], result[2] = recall, precision, ndcg
    print('\t@10\t@20\t@50')
    print('Rec:\t{:.4f}\t{:.4f}\t{:.4f}\nPre:\t{:.4f}\t{:.4f}\t{:.4f}\nNDCG:\t{:.4f}\t{:.4f}\t{:.4f}'.format(
        result[0,0], result[0,1], result[0,2], result[1,0], result[1,1], result[1,2], result[2,0], result[2,1], result[2,2]
    ))
    return result


# calculate all metrics (precision/recall/ndcg + sp/eo/nsp/neo)
def New_Eval(args, Rec, eval_dict, exclude_list, item_groups, user_neighbors, K=20):
    preK, recK, ndcgK = np.zeros(3)
    user_exclude_indices, item_exclude_indices = np.ones_like(Rec), np.zeros((args.num_group, Rec.shape[1]))
    num_user = len(eval_dict)
    item_groups, user_neighbors = np.array(item_groups), np.array(user_neighbors)
    sp_scores, eo_scores, nsp_scores, neo_scores = [[[] for _ in range(args.num_group)] for _ in range(4)]
    Rec = sigmoid(Rec)
    for u in range(Rec.shape[0]):
        like_item = np.array(exclude_list[u])
        Rec[u, like_item] = 0
        user_exclude_indices[u, like_item] = 0
    for idx in range(args.num_group):
        item_exclude_indices[idx] = (item_groups==idx).astype(int)

    for u in eval_dict:  # iterate each user
        u_test, heap, u_exclude, utility = eval_dict[u], Rec[u], user_exclude_indices[u], user_neighbors[u]
        topK = heapq.nlargest(K, range(len(heap)), heap.__getitem__)
        precision_u, recall_u, ndcg_u = New_Eval_User(topK, u_test, K)
        preK += precision_u
        recK += recall_u
        ndcgK += ndcg_u
        for idx in range(args.num_group):  # SP, NSP
            user_group_scores = Rec[u,(u_exclude*item_exclude_indices[idx]).astype(bool)]
            sp_scores[idx].extend(list(user_group_scores))
            nsp_scores[idx].extend(list(user_group_scores*utility))
        for test_i in u_test:  # EO, NEO
            group = item_groups[test_i]
            eo_scores[group].append(Rec[u,test_i])
            neo_scores[group].append(Rec[u,test_i]*utility)
    sp_list, eo_list = [np.mean(score_list) for score_list in sp_scores], [np.mean(score_list) for score_list in eo_scores]
    nsp_list, neo_list = [np.mean(score_list) for score_list in nsp_scores], [np.mean(score_list) for score_list in neo_scores]
    return preK/num_user, recK/num_user, ndcgK/num_user, sp_list, nsp_list, eo_list, neo_list


def New_Eval_User(topK, test, K=20):
    dcg_list, count_K = [], 0
    for i in range(K):
        hit = int(topK[i] in test)
        count_K += hit
        dcg_list.append(hit)
    idcg_list = np.ones(len(test)).tolist()
    ndcg_K = NDCG_at_k(dcg_list, idcg_list, K)
    precision_K = count_K / K
    recall_K = count_K / len(test)
    return precision_K, recall_K, ndcg_K


def Display_New_Eval(args, model, eval_dict, exclude_list, item_groups, user_neighbors, K=20):
    Rec = model.predict().detach().cpu().numpy()
    pre, rec, ndcg, sp, nsp, eo, neo, rsp, nrsp, reo, nreo = New_Eval(args, Rec, eval_dict, exclude_list, item_groups, user_neighbors, K=20, return_list=False)
    print('Pre\tRecall\tNDCG\tSP\tNSP\tEO\tNEO\tRSP\tNRSP\tREO\tNREO')
    print('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(
        pre, rec, ndcg, sp, nsp, eo, neo, rsp, nrsp, reo, nreo))
    print('SPD: {:.2f}\tEOD: {:.2f}\tRSPD: {:.2f}\tREOD: {:.2f}\t'.format(deg([sp, nsp]), deg([eo, neo]), deg([rsp, nrsp]), deg([reo, nreo])))
    return pre, rec, ndcg, sp, nsp, eo, neo, rsp, nrsp, reo, nreo


def calculate_noranking_fairness(args, Rec, eval_dict, exclude_list, item_groups, user_neighbors):
    Rec = sigmoid(Rec)
    item_groups, user_neighbors = np.array(item_groups), np.array(user_neighbors)
    user_exclude_indices, item_exclude_indices = np.ones_like(Rec), np.zeros((args.num_group, Rec.shape[1]))
    for u in range(Rec.shape[0]):
        like_item = np.array(exclude_list[u])
        Rec[u, like_item] = 0
        user_exclude_indices[u, like_item] = 0
    for idx in range(args.num_group):
        item_exclude_indices[idx] = (item_groups==idx).astype(int)

    if args.mode in {'sp','nsp','sp_nsp'}:
        sp_scores, nsp_scores = [[] for _ in range(args.num_group)], [[] for _ in range(args.num_group)]
        for u in eval_dict:
            u_exclude, utility = user_exclude_indices[u], user_neighbors[u]
            for idx in range(args.num_group):
                user_group_scores = Rec[u,(u_exclude*item_exclude_indices[idx]).astype(bool)]
                sp_scores[idx].extend(list(user_group_scores))
                nsp_scores[idx].extend(list(user_group_scores*utility))
        sp_list, nsp_list = [np.mean(score_list) for score_list in sp_scores], [np.mean(score_list) for score_list in nsp_scores]
        return sp_list, nsp_list

    elif args.mode in {'eo','neo','eo_neo'}:
        eo_scores, neo_scores = [[] for _ in range(args.num_group)], [[] for _ in range(args.num_group)]
        for u in eval_dict:
            u_test, utility = eval_dict[u], user_neighbors[u]
            for test_i in u_test:
                group = item_groups[test_i]
                eo_scores[group].append(Rec[u,test_i])
                neo_scores[group].append(Rec[u,test_i]*utility)
        eo_list, neo_list = [np.mean(score_list) for score_list in eo_scores], [np.mean(score_list) for score_list in neo_scores]
        return eo_list, neo_list


def display_noranking_results(args, prob_list, nprob_list):
    speo, nspneo = rsd(prob_list), rsd(nprob_list)
    print('{}={:.4f}\tn{}={:.4f}\tdeg={:.2f}\t'.format(args.mode, speo, args.mode, nspneo, deg([speo, nspneo])))
    return speo, nspneo


def display_all_results(ndcg, sp_list, nsp_list, eo_list, neo_list):
    sp, eo, nsp, neo = rsd(sp_list), rsd(eo_list), rsd(nsp_list), rsd(neo_list)
    f1sp, degsp, f1eo, degeo = f1(sp,nsp), deg([sp,nsp]), f1(eo,neo), deg([eo,neo])
    print('NDCG\tSP\tNSP\tF1SP\tdegSP\tEO\tNEO\tF1EO\tdegEO')
    print('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.2f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.2f}'.format(ndcg, sp, nsp, f1sp, degsp, eo, neo, f1eo, degeo))
