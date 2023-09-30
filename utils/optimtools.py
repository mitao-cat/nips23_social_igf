import numpy as np
import torch
import torch.utils.data
from torch.autograd import Variable


def circle_points(r, n):
    """
    generate evenly distributed unit preference vectors for two tasks
    """
    circles = []
    for r, n in zip(r, n):
        t = np.linspace(0, 0.5 * np.pi, n)
        x = r * np.cos(t)
        y = r * np.sin(t)
        circles.append(np.c_[x, y])
    return circles


def get_preference_vectors(npref):
    vectors = np.zeros((npref,2))
    for i in range(npref):
        t = (i+0.5) * np.pi/(2*npref)
        vectors[i] = np.array([np.cos(t),np.sin(t)])
    return vectors


# 0417_0418 version
def batch_fairness(args, users, item_groups, user_neighbors, pos_items, neg_items, pos_scores, neg_scores):
    item_groups, user_neighbors = torch.LongTensor(item_groups).to(args.device), torch.LongTensor(user_neighbors).to(args.device)
    pos_items_groups, neg_items_groups, user_utilities = item_groups[pos_items], item_groups[neg_items], user_neighbors[users]
    # pos_items_groups, neg_items_groups, user_utilities = item_groups[pos_items], item_groups[neg_items], user_neighbors[users]+1
    sp_pt, nsp_pt, eo_pt, neo_pt, speo_pt, nspeo_pt = [torch.zeros(args.num_group).to(args.device) for _ in range(6)]
    pos_prob, neg_prob = torch.sigmoid(pos_scores), torch.sigmoid(neg_scores)
    
    for idx in range(args.num_group):
        pos_indices, neg_indices = pos_items_groups==idx, neg_items_groups==idx
        selected_pos_items, selected_neg_items = pos_items[pos_indices], neg_items[neg_indices]
        selected_pos_scores, selected_neg_scores = pos_prob[pos_indices], neg_prob[neg_indices]
        pos_user_utilities, neg_user_utilities = user_utilities[pos_indices], user_utilities[neg_indices]

        speo = torch.mean(torch.cat((selected_pos_scores, selected_neg_scores)))
        nspeo = torch.mean(torch.cat((selected_pos_scores*pos_user_utilities, selected_neg_scores*neg_user_utilities)))
        sp, nsp = torch.mean(selected_neg_scores), torch.mean(selected_neg_scores*neg_user_utilities)
        eo, neo = torch.mean(selected_pos_scores), torch.mean(selected_pos_scores*pos_user_utilities)
        sp_pt[idx], nsp_pt[idx], eo_pt[idx], neo_pt[idx], speo_pt[idx], nspeo_pt[idx] = sp, nsp, eo, neo, speo, nspeo
    
    if args.mode == 'sp_eo':
        return speo_pt, nspeo_pt
    elif args.mode == 'sp':
        return sp_pt, nsp_pt
    elif args.mode == 'eo':
        return eo_pt, neo_pt
    raise NotImplementedError


def batch_group_probs(args, users, item_groups, user_neighbors, pos_items, neg_items, pos_scores, neg_scores):
    user_utilities = user_neighbors[users]
    if args.mode == 'eo':
        eo_pt, neo_pt = torch.zeros(args.num_group).to(args.device), torch.zeros(args.num_group).to(args.device)
        pos_items_groups, pos_prob = item_groups[pos_items], torch.sigmoid(pos_scores)
        for idx in range(args.num_group):
            pos_indices = pos_items_groups==idx
            selected_pos_items, selected_pos_scores, pos_user_utilities = pos_items[pos_indices], pos_prob[pos_indices], user_utilities[pos_indices]
            eo_pt[idx], neo_pt[idx] = torch.mean(selected_pos_scores), torch.mean(selected_pos_scores*pos_user_utilities)
        return eo_pt, neo_pt
    elif args.mode == 'sp':
        sp_pt, nsp_pt = torch.zeros(args.num_group).to(args.device), torch.zeros(args.num_group).to(args.device)
        neg_items_groups, neg_prob = item_groups[neg_items], torch.sigmoid(neg_scores)
        for idx in range(args.num_group):
            neg_indices = neg_items_groups==idx
            selected_neg_items, selected_neg_scores, neg_user_utilities = neg_items[neg_indices], neg_prob[neg_indices], user_utilities[neg_indices]
            sp_pt[idx], nsp_pt[idx] = torch.mean(selected_neg_scores), torch.mean(selected_neg_scores*neg_user_utilities)
        return sp_pt, nsp_pt


def get_d_paretomtl_init(grads,value,weights,i):
    """ 
    calculate the gradient direction for ParetoMTL initialization 
    """
    
    flag = False
    nobj = value.shape
   
    # check active constraints
    current_weight = weights[i]
    rest_weights = weights
    w = rest_weights - current_weight
    
    gx =  torch.matmul(w,value/torch.norm(value))
    idx = gx >  0
   
    # calculate the descent direction
    if torch.sum(idx) <= 0:
        flag = True
        return flag, torch.zeros(nobj)
    if torch.sum(idx) == 1:
        sol = torch.ones(1).cuda().float()
    else:
        vec =  torch.matmul(w[idx],grads)
        sol, nd = MinNormSolver.find_min_norm_element([[vec[t]] for t in range(len(vec))])


    weight0 =  torch.sum(torch.stack([sol[j] * w[idx][j ,0] for j in torch.arange(0, torch.sum(idx))]))
    weight1 =  torch.sum(torch.stack([sol[j] * w[idx][j ,1] for j in torch.arange(0, torch.sum(idx))]))
    weight = torch.stack([weight0,weight1])
   
    
    return flag, weight


def get_d_paretomtl_recloss(grads, losses_vec, ref_vec, pref_idx, recloss, thre):
    """ calculate the gradient direction for ParetoMTL """
    current_weight = ref_vec[pref_idx]  # check active constraints
    w = ref_vec - current_weight
    gx =  torch.matmul(w, losses_vec / torch.norm(losses_vec))
    idx = gx > 0

    if recloss.detach().cpu().item() <= thre:   # without consideration of recommendation loss
        grads = grads[:-1]
        if torch.sum(idx) <= 0:
            sol, nd = MinNormSolver.find_min_norm_element([[grads[t]] for t in range(len(grads))])
            return torch.tensor(sol).cuda().float()
        vec = torch.cat((grads, torch.matmul(w[idx], grads)))
        sol, nd = MinNormSolver.find_min_norm_element([[vec[t]] for t in range(len(vec))])
        weight0 = sol[0] + torch.sum(torch.stack([sol[j]*w[idx][j-2,0] for j in torch.arange(2, 2+torch.sum(idx))]))
        weight1 = sol[1] + torch.sum(torch.stack([sol[j]*w[idx][j-2,1] for j in torch.arange(2, 2+torch.sum(idx))]))
        weight = torch.stack([weight0, weight1])
        return weight
    else:   # take recommendation loss into consideration
        if torch.sum(idx) <= 0:
            sol, nd = MinNormSolver.find_min_norm_element([[grads[t]] for t in range(len(grads))])
            return torch.tensor(sol).cuda().float()
        vec = torch.cat((grads, torch.matmul(w[idx], grads[:-1])))
        sol, nd = MinNormSolver.find_min_norm_element([[vec[t]] for t in range(len(vec))])
        weight0 = sol[0] + torch.sum(torch.stack([sol[j]*w[idx][j-3,0] for j in torch.arange(3, 3+torch.sum(idx))]))
        weight1 = sol[1] + torch.sum(torch.stack([sol[j]*w[idx][j-3,1] for j in torch.arange(3, 3+torch.sum(idx))]))
        weight = torch.stack([weight0, weight1, torch.tensor(sol[2]).to(w.device)])
        return weight


def get_d_paretomtl_wo_recloss(grads, losses_vec, ref_vec, pref_idx):
    """ calculate the gradient direction for ParetoMTL """
    current_weight = ref_vec[pref_idx]  # check active constraints
    w = ref_vec - current_weight
    gx =  torch.matmul(w, losses_vec / torch.norm(losses_vec))
    idx = gx > 0
    if torch.sum(idx) <= 0:
        sol, nd = MinNormSolver.find_min_norm_element([[grads[t]] for t in range(len(grads))])
        return torch.tensor(sol).cuda().float()
    
    vec = torch.cat((grads, torch.matmul(w[idx], grads)))
    sol, nd = MinNormSolver.find_min_norm_element([[vec[t]] for t in range(len(vec))])
    weight0 = sol[0] + torch.sum(torch.stack([sol[j]*w[idx][j-2,0] for j in torch.arange(2, 2+torch.sum(idx))]))
    weight1 = sol[1] + torch.sum(torch.stack([sol[j]*w[idx][j-2,1] for j in torch.arange(2, 2+torch.sum(idx))]))
    weight = torch.stack([weight0, weight1])
    return weight


def get_d_moomtl_wo_recloss(grads):
    """ calculate the gradient direction for MOOMTL """
    sol, nd = MinNormSolver.find_min_norm_element([[grads[t]] for t in range(len(grads))])
    return torch.tensor(sol).cuda().float()


def get_d_moomtl(grads, recloss, thre):
    if recloss.detach().cpu().item() <= thre:   # without consideration of recommendation loss
        grads = grads[:-1]
    sol, nd = MinNormSolver.find_min_norm_element([[grads[t]] for t in range(len(grads))])
    return torch.tensor(sol).cuda().float()


def single_batch_group_probs(args, users, item_groups, user_neighbors, pos_items, neg_items, pos_scores, neg_scores):
    user_utilities = user_neighbors[users]
    if args.mode in {'eo','neo'}:
        eo_pt, neo_pt = torch.zeros(args.num_group).to(args.device), torch.zeros(args.num_group).to(args.device)
        pos_items_groups, pos_prob = item_groups[pos_items], torch.sigmoid(pos_scores)
        for idx in range(args.num_group):
            pos_indices = pos_items_groups==idx
            selected_pos_items, selected_pos_scores, pos_user_utilities = pos_items[pos_indices], pos_prob[pos_indices], user_utilities[pos_indices]
            eo_pt[idx], neo_pt[idx] = torch.mean(selected_pos_scores), torch.mean(selected_pos_scores*pos_user_utilities)
        if args.mode == 'eo':
            return eo_pt
        return neo_pt
    elif args.mode in {'sp','nsp'}:
        sp_pt, nsp_pt = torch.zeros(args.num_group).to(args.device), torch.zeros(args.num_group).to(args.device)
        neg_items_groups, neg_prob = item_groups[neg_items], torch.sigmoid(neg_scores)
        for idx in range(args.num_group):
            neg_indices = neg_items_groups==idx
            selected_neg_items, selected_neg_scores, neg_user_utilities = neg_items[neg_indices], neg_prob[neg_indices], user_utilities[neg_indices]
            sp_pt[idx], nsp_pt[idx] = torch.mean(selected_neg_scores), torch.mean(selected_neg_scores*neg_user_utilities)
        if args.mode == 'sp':
            return sp_pt
        return nsp_pt


class MinNormSolver:
    MAX_ITER = 250
    STOP_CRIT = 1e-5

    def _min_norm_element_from2(v1v1, v1v2, v2v2):
        """
        Analytical solution for min_{c} |cx_1 + (1-c)x_2|_2^2
        d is the distance (objective) optimzed
        v1v1 = <x1,x1>
        v1v2 = <x1,x2>
        v2v2 = <x2,x2>
        """
        if v1v2 >= v1v1:
            # Case: Fig 1, third column
            gamma = 0.999
            cost = v1v1
            return gamma, cost
        if v1v2 >= v2v2:
            # Case: Fig 1, first column
            gamma = 0.001
            cost = v2v2
            return gamma, cost
        # Case: Fig 1, second column
        gamma = -1.0 * ( (v1v2 - v2v2) / (v1v1+v2v2 - 2*v1v2) )
        cost = v2v2 + gamma*(v1v2 - v2v2)
        return gamma, cost

    def _min_norm_2d(vecs, dps):
        """
        Find the minimum norm solution as combination of two points
        This is correct only in 2D
        ie. min_c |\sum c_i x_i|_2^2 st. \sum c_i = 1 , 1 >= c_1 >= 0 for all i, c_i + c_j = 1.0 for some i, j
        """
        dmin = 1e8
        for i in range(len(vecs)):
            for j in range(i+1,len(vecs)):
                if (i,j) not in dps:
                    dps[(i, j)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(i,j)] += torch.dot(vecs[i][k], vecs[j][k]).item()#torch.dot(vecs[i][k], vecs[j][k]).data[0]
                    dps[(j, i)] = dps[(i, j)]
                if (i,i) not in dps:
                    dps[(i, i)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(i,i)] += torch.dot(vecs[i][k], vecs[i][k]).item()#torch.dot(vecs[i][k], vecs[i][k]).data[0]
                if (j,j) not in dps:
                    dps[(j, j)] = 0.0   
                    for k in range(len(vecs[i])):
                        dps[(j, j)] += torch.dot(vecs[j][k], vecs[j][k]).item()#torch.dot(vecs[j][k], vecs[j][k]).data[0]
                c,d = MinNormSolver._min_norm_element_from2(dps[(i,i)], dps[(i,j)], dps[(j,j)])
                if d < dmin:
                    dmin = d
                    sol = [(i,j),c,d]
        return sol, dps

    def _projection2simplex(y):
        """
        Given y, it solves argmin_z |y-z|_2 st \sum z = 1 , 1 >= z_i >= 0 for all i
        """
        m = len(y)
        sorted_y = np.flip(np.sort(y), axis=0)
        tmpsum = 0.0
        tmax_f = (np.sum(y) - 1.0)/m
        for i in range(m-1):
            tmpsum+= sorted_y[i]
            tmax = (tmpsum - 1)/ (i+1.0)
            if tmax > sorted_y[i+1]:
                tmax_f = tmax
                break
        return np.maximum(y - tmax_f, np.zeros(y.shape))
    
    def _next_point(cur_val, grad, n):
        proj_grad = grad - ( np.sum(grad) / n )
        tm1 = -1.0*cur_val[proj_grad<0]/proj_grad[proj_grad<0]
        tm2 = (1.0 - cur_val[proj_grad>0])/(proj_grad[proj_grad>0])
        
        skippers = np.sum(tm1<1e-7) + np.sum(tm2<1e-7)
        t = 1
        if len(tm1[tm1>1e-7]) > 0:
            t = np.min(tm1[tm1>1e-7])
        if len(tm2[tm2>1e-7]) > 0:
            t = min(t, np.min(tm2[tm2>1e-7]))

        next_point = proj_grad*t + cur_val
        next_point = MinNormSolver._projection2simplex(next_point)
        return next_point

    def find_min_norm_element(vecs):
        """
        Given a list of vectors (vecs), this method finds the minimum norm element in the convex hull
        as min |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
        It is quite geometric, and the main idea is the fact that if d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j; the solution lies in (0, d_{i,j})
        Hence, we find the best 2-task solution, and then run the projected gradient descent until convergence
        """
        # Solution lying at the combination of two points
        dps = {}
        init_sol, dps = MinNormSolver._min_norm_2d(vecs, dps)
        
        n=len(vecs)
        sol_vec = np.zeros(n)
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if n < 3:
            # This is optimal for n=2, so return the solution
            return sol_vec , init_sol[2]
    
        iter_count = 0

        grad_mat = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                grad_mat[i,j] = dps[(i, j)]
                

        while iter_count < MinNormSolver.MAX_ITER:
            grad_dir = -1.0*np.dot(grad_mat, sol_vec)
            new_point = MinNormSolver._next_point(sol_vec, grad_dir, n)
            # Re-compute the inner products for line search
            v1v1 = 0.0
            v1v2 = 0.0
            v2v2 = 0.0
            for i in range(n):
                for j in range(n):
                    v1v1 += sol_vec[i]*sol_vec[j]*dps[(i,j)]
                    v1v2 += sol_vec[i]*new_point[j]*dps[(i,j)]
                    v2v2 += new_point[i]*new_point[j]*dps[(i,j)]
            nc, nd = MinNormSolver._min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc*sol_vec + (1-nc)*new_point
            change = new_sol_vec - sol_vec
            if np.sum(np.abs(change)) < MinNormSolver.STOP_CRIT:
                return sol_vec, nd
            sol_vec = new_sol_vec

    def find_min_norm_element_FW(vecs):
        """
        Given a list of vectors (vecs), this method finds the minimum norm element in the convex hull
        as min |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
        It is quite geometric, and the main idea is the fact that if d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j; the solution lies in (0, d_{i,j})
        Hence, we find the best 2-task solution, and then run the Frank Wolfe until convergence
        """
        # Solution lying at the combination of two points
        dps = {}
        init_sol, dps = MinNormSolver._min_norm_2d(vecs, dps)

        n=len(vecs)
        sol_vec = np.zeros(n)
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if n < 3:
            # This is optimal for n=2, so return the solution
            return sol_vec , init_sol[2]

        iter_count = 0

        grad_mat = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                grad_mat[i,j] = dps[(i, j)]

        while iter_count < MinNormSolver.MAX_ITER:
            t_iter = np.argmin(np.dot(grad_mat, sol_vec))

            v1v1 = np.dot(sol_vec, np.dot(grad_mat, sol_vec))
            v1v2 = np.dot(sol_vec, grad_mat[:, t_iter])
            v2v2 = grad_mat[t_iter, t_iter]

            nc, nd = MinNormSolver._min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc*sol_vec
            new_sol_vec[t_iter] += 1 - nc

            change = new_sol_vec - sol_vec
            if np.sum(np.abs(change)) < MinNormSolver.STOP_CRIT:
                return sol_vec, nd
            sol_vec = new_sol_vec
