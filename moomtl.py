import torch
import time
import os
from utils.utils import parse_args, setup_seed, random_sampler
from utils.models import BaseMF
from utils.dataloader import get_dataloader, get_data
from utils.evaluate import calculate_noranking_fairness, New_Eval, deg, display_all_results, display_noranking_results, f1
from utils.optimtools import batch_group_probs, get_d_moomtl_wo_recloss
from torch.autograd import Variable


class EarlyStopping:    # different from EarlyStopping from utils.utils!
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_score = 99
        self.early_stop = False
        self.best_state = None

    def __call__(self, score, model):
        if score <= self.best_score:
            self.save_checkpoint(score, model)
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop
        
    def save_checkpoint(self, score, model):
        self.best_state = {key: value.cpu() for key, value in model.state_dict().items()}                
        self.best_score = score
        self.counter = 0


def fitlog_addloss(args, step, fitlog, rsd, rsdn, std, stdn, avg, avgn):
    mode = args.mode    # {sp,eo}
    fitlog.add_loss(rsd, name=mode, step=step)
    fitlog.add_loss(rsdn, name='n'+mode, step=step)
    fitlog.add_loss(deg([rsd, rsdn]), name="Deg", step=step)
    fitlog.add_loss(std, name="std_"+mode, step=step)
    fitlog.add_loss(stdn, name="std_n"+mode, step=step)
    fitlog.add_loss(avg, name="avg_"+mode, step=step)
    fitlog.add_loss(avgn, name="avg_n"+mode, step=step)


def train_model(item_groups, user_neighbors):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    earlystopper = EarlyStopping(patience=args.patience)    # default: 5
    model.eval()
    Rec = model.predict().detach().cpu().numpy()
    prob_list, nprob_list = calculate_noranking_fairness(args, Rec, val_dict, train_list, item_groups, user_neighbors)
    _, _ = display_noranking_results(args, prob_list, nprob_list)
    print('----- Running the MOOMTL -----')
    for epoch in range(1, args.max_epoch+1):
        print('********** Epoch {} **********'.format(epoch))
        epoch_start_time = time.time()
        total_neg_sample_time, total_train_inference_time, total_pareto_time, total_bprloss, total_regloss = 0, 0, 0, 0, 0
        item_groups, user_neighbors = torch.LongTensor(item_groups).to(device), torch.LongTensor(user_neighbors).to(device)
        model.train()
        for _, (users, pos_items) in enumerate(train_loader):
            sample_start_time = time.time()
            neg_items = random_sampler(users.numpy(), pos_items.numpy(), train_list, args)
            sample_end_time = time.time()
            users, pos_items, neg_items = users.to(device), pos_items.to(device), neg_items.to(device)
            pos_scores, neg_scores = model.calculate_pos_neg_scores(users, pos_items, neg_items)
            bprloss, regloss = model.calculate_loss(pos_scores, neg_scores)
            total_bprloss += bprloss.detach().cpu().item()
            total_regloss += args.reg*regloss.detach().cpu().item()

            pareto_start_time = time.time()
            grads = {0:[],1:[]}
            speo_pt, nspneo_pt = batch_group_probs(args, users, item_groups, user_neighbors, pos_items, neg_items, pos_scores, neg_scores)
            rsd_speo, rsd_nspneo = torch.std(speo_pt) / torch.mean(speo_pt), torch.std(nspneo_pt) / torch.mean(nspneo_pt)
            optimizer.zero_grad()   # SP/EO
            rsd_speo.backward(retain_graph=True)
            for param in model.parameters():
                if param.grad is not None:
                    grads[0].append(Variable(param.grad.data.clone().flatten(), requires_grad=False))
            optimizer.zero_grad()   # NSP/NEO
            rsd_nspneo.backward(retain_graph=True)
            for param in model.parameters():
                if param.grad is not None:
                    grads[1].append(Variable(param.grad.data.clone().flatten(), requires_grad=False))
            grads = torch.stack([torch.cat(grads[i]) for i in range(2)])
            weight_vec = get_d_moomtl_wo_recloss(grads)
            normalize_coeff = n_tasks / torch.sum(torch.abs(weight_vec))
            weight_vec = weight_vec * normalize_coeff
            pareto_end_time = time.time()

            optimizer.zero_grad()
            loss_total = weight_vec[0] * rsd_speo + weight_vec[1] * rsd_nspneo
            loss_total.backward()
            optimizer.step()
            total_train_inference_time += (pareto_start_time + time.time() - sample_end_time - pareto_end_time)
            total_pareto_time += (pareto_end_time - pareto_start_time)
            total_neg_sample_time += (sample_end_time - sample_start_time)
        avgbpr, avgreg, avgloss = total_bprloss / len(train_loader), total_regloss / len(train_loader), (total_regloss+total_bprloss) / len(train_loader)
        print('Time:{:.2f}s = sample_{:.2f}s + trainfer_{:.2f} + pareto_{:.2f}s\tAvgRecLoss:{:.4f} = bpr_{:.4f} + reg_{:.4f}'.format(
            time.time()-epoch_start_time, total_neg_sample_time, total_train_inference_time, total_pareto_time, avgloss, avgbpr, avgreg
        ))
        model.eval()
        Rec = model.predict().detach().cpu().numpy()
        item_groups, user_neighbors = item_groups.cpu().numpy().tolist(), user_neighbors.cpu().numpy().tolist()
        prob_list, nprob_list = calculate_noranking_fairness(args, Rec, val_dict, train_list, item_groups, user_neighbors)
        rsd, rsdn = display_noranking_results(args, prob_list, nprob_list)
        f1value = f1(rsd,rsdn)
        if earlystopper(f1value, model) is True:
            break
    print('Loading {}th epoch'.format(min(epoch-args.patience, args.max_epoch)))
    model.load_state_dict(earlystopper.best_state)
    model.eval()
    Rec = model.predict().detach().cpu().numpy()
    print('********** Validating **********')
    _, _, ndcg, sp_list, nsp_list, eo_list, neo_list = New_Eval(args, Rec, val_dict, train_list, item_groups, user_neighbors, K=args.K)
    display_all_results(ndcg, sp_list, nsp_list, eo_list, neo_list)
    print('********** Testing **********')
    _, _, ndcg, sp_list, nsp_list, eo_list, neo_list = New_Eval(args, Rec, test_dict, trainval_list, item_groups, user_neighbors, K=args.K)
    display_all_results(ndcg, sp_list, nsp_list, eo_list, neo_list)


if __name__ == "__main__":
    args = parse_args()
    setup_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device, args.reg = device, args.reg / 2
    n_tasks, npref, pref_idx = 2, args.npref, args.pref_idx
    assert args.pref_idx < args.npref
    assert args.mode in {'sp','eo'}
    assert args.dataset in {'KuaiRec','Epinions'}

    train_dict, val_dict, test_dict, num_user, num_item, num_train, _, _, item_groups, user_neighbors, _, _ = get_data(args.dataset)
    args.num_user, args.num_item, args.num_train, args.num_group = num_user, num_item, num_train, max(item_groups)+1
    train_loader, train_list, trainval_list = get_dataloader(args, train_dict, val_dict)
    model = BaseMF(args).to(device)
    model.load_state_dict(torch.load('./data/{}/best_model.pth.tar'.format(args.dataset)))
    train_model(item_groups, user_neighbors)
