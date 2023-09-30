import torch
import time
from utils.utils import parse_args, setup_seed, random_sampler
from utils.models import BaseMF
from utils.dataloader import get_dataloader, get_data
from utils.evaluate import New_Eval, display_all_results, f1, calculate_noranking_fairness, display_noranking_results
from utils.optimtools import batch_group_probs, single_batch_group_probs


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


def train_model(item_groups, user_neighbors):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    earlystopper = EarlyStopping(patience=args.patience)
    mode = args.mode
    if args.mode in {'sp_nsp','eo_neo'}:
        args.mode = args.mode[:2]   # sp eo
    model.eval()
    Rec = model.predict().detach().cpu().numpy()
    prob_list, nprob_list = calculate_noranking_fairness(args, Rec, val_dict, train_list, item_groups, user_neighbors)
    _, _ = display_noranking_results(args, prob_list, nprob_list)

    print('----- Running regularization baseline. -----')
    for epoch in range(1, args.max_epoch+1):
        print('********** Epoch {} **********'.format(epoch))
        epoch_start_time = time.time()
        item_groups, user_neighbors = torch.LongTensor(item_groups).to(device), torch.LongTensor(user_neighbors).to(device)
        total_bprloss, total_recreg_loss, total_reg1_loss, total_reg2_loss = 0, 0, 0, 0
        total_neg_sample_time, total_train_time = 0, 0
        model.train()
        for _, (users, pos_items) in enumerate(train_loader):
            sample_start_time = time.time()
            neg_items = random_sampler(users.numpy(), pos_items.numpy(), train_list, args)
            sample_end_time = time.time()
            users, pos_items, neg_items = users.to(device), pos_items.to(device), neg_items.to(device)
            pos_scores, neg_scores = model.calculate_pos_neg_scores(users, pos_items, neg_items)
            bprloss, regloss = model.calculate_loss(pos_scores, neg_scores)
            total_bprloss += bprloss.detach().cpu().item()
            total_recreg_loss += args.reg * regloss.detach().cpu().item()

            if mode in {'sp','eo','nsp','neo'}:
                prob_pt = single_batch_group_probs(args, users, item_groups, user_neighbors, pos_items, neg_items, pos_scores, neg_scores)
                rsd = torch.std(prob_pt) / torch.mean(prob_pt)
                total_reg1_loss += args.reg1 * rsd.detach().cpu().item()
                loss = bprloss + args.reg*regloss + args.reg1*rsd
            elif mode in {'sp_nsp','eo_neo'}:
                sp_pt, nsp_pt = batch_group_probs(args, users, item_groups, user_neighbors, pos_items, neg_items, pos_scores, neg_scores)
                rsd_sp, rsd_nsp = torch.std(sp_pt) / torch.mean(sp_pt), torch.std(nsp_pt) / torch.mean(nsp_pt)
                total_reg1_loss += args.reg1 * rsd_sp.detach().cpu().item()
                total_reg2_loss += args.reg2 * rsd_nsp.detach().cpu().item()
                loss = bprloss + args.reg*regloss + args.reg1*rsd_sp + args.reg2*rsd_nsp
            loss.backward()
            optimizer.step()
            total_neg_sample_time += (sample_end_time - sample_start_time)
            total_train_time += (time.time() - sample_end_time)
        
        nb = len(train_loader)
        avgbpr, avgrecreg, avgreg1, avgreg2 = total_bprloss/nb, total_recreg_loss/nb, total_reg1_loss/nb, total_reg2_loss/nb
        avgloss = avgbpr + avgrecreg + avgreg1 + avgreg2
        print('Time:{:.2f}s = sample_{:.2f}s + train_{:.2f}s\tAvgLoss:{:.4f} = bpr_{:.4f} + reg_{:.4f} + {}_{:.4f} + {}_{:.4f}'.format(
            time.time()-epoch_start_time, total_neg_sample_time, total_train_time, 
            avgloss, avgbpr, avgrecreg, args.mode, avgreg1, 'n'+args.mode, avgreg2))
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
    Rec = model.predict().cpu().numpy()
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
    args.device, args.reg, args.max_epoch = device, args.reg/2, 200
    assert args.mode in {'sp','eo','nsp','neo','sp_nsp','eo_neo'}
    assert args.dataset in {'KuaiRec','Epinions'}
    train_dict, val_dict, test_dict, num_user, num_item, num_train, _, _, item_groups, user_neighbors, _, _ = get_data(args.dataset)
    args.num_user, args.num_item, args.num_train, args.num_group = num_user, num_item, num_train, max(item_groups)+1
    train_loader, train_list, trainval_list = get_dataloader(args, train_dict, val_dict)
    model = BaseMF(args).to(device)
    model.load_state_dict(torch.load('./data/{}/best_model.pth.tar'.format(args.dataset)))
    train_model(item_groups, user_neighbors)
