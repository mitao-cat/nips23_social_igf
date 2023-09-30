import pickle
import numpy as np
import torch
import torch.utils.data as Data


def get_data(dataset):
    root_dir = './data/{}/'.format(dataset)
    train_dict = pickle.load(open(root_dir+'train_dict.pkl', 'rb'))
    val_dict = pickle.load(open(root_dir+'val_dict.pkl', 'rb'))
    test_dict = pickle.load(open(root_dir+'test_dict.pkl', 'rb'))
    item_categories_list = pickle.load(open(root_dir+'item_categories_list.pkl', 'rb'))
    user_num_fans_list = pickle.load(open(root_dir+'user_num_fans_list.pkl', 'rb'))
    category_item_dict = pickle.load(open(root_dir+'category_item_dict.pkl', 'rb'))
    category_num_dict = pickle.load(open(root_dir+'category_num_dict.pkl', 'rb'))

    num_train = sum([len(l) for l in train_dict.values()])
    num_val = sum([len(l) for l in val_dict.values()])
    num_test = sum([len(l) for l in test_dict.values()])
    num_user, num_item = len(train_dict), len(item_categories_list)

    sorted_categories = sorted(list(category_item_dict.keys()))
    cate_idx_dict, new_category_item_dict, new_category_num_dict = {}, {}, {}
    for idx, cate in enumerate(sorted_categories):
        cate_idx_dict[cate] = idx
    new_item_categories_list = [cate_idx_dict[cate] for cate in item_categories_list]
    for cate in cate_idx_dict:
        new_category_item_dict[cate_idx_dict[cate]] = category_item_dict[cate]
        new_category_num_dict[cate_idx_dict[cate]] = category_num_dict[cate]

    del item_categories_list, category_item_dict, category_num_dict
    print(num_user, num_item, num_train, num_val, num_test)
    return train_dict, val_dict, test_dict, num_user, num_item, num_train, num_val, num_test, \
    new_item_categories_list, user_num_fans_list, new_category_item_dict, new_category_num_dict


def get_dataloader(args, train_dict, val_dict):
    num_train, numneg = args.num_train, args.numneg
    users, items = np.zeros(num_train*numneg), np.zeros(num_train*numneg)
    index = 0
    train_list, trainval_list = [], []
    for u in train_dict:
        num_user_train = len(train_dict[u])*numneg
        users[index:index+num_user_train] = np.ones(num_user_train)*u
        items[index:index+num_user_train] = np.array(train_dict[u]*numneg)
        train_list.append(train_dict[u])
        if u not in val_dict:
            trainval_list.append(train_dict[u])
        else:
            trainval_list.append(train_dict[u]+val_dict[u])
        index += num_user_train
    indices = np.arange(len(users))
    np.random.shuffle(indices)
    users, items = torch.LongTensor(users[indices]), torch.LongTensor(items[indices])
    tensor_dataset = Data.TensorDataset(users, items)
    loader = Data.DataLoader(dataset=tensor_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    return loader, train_list, trainval_list
