# @Time   : 2023.03.03
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
import numpy as np
import torch
from lib import json_util as ju
from lib import callback

def cal_sparsity(u_num, req_types, req_num):
    '''Calculate the sparsity of the dataset

    Parameters:
    -----------

    u_num: int
    number of users

    req_types: int
    number of requests's type

    req_num: int
    Total number of requests for all users

    Returns:
    --------

    sparsity: float
    '''
    return 1 - req_num/u_num/req_types;

def general_pcr(metadata):
    '''general purpose processor

    Parameters:
    -----------

    metadata: np.narray
    raw data to be processed
    format:[[userid, itemid, t]]

    Returns:
    --------
    udataset: dict
    structure
    --- u_num
     |- req_num
     |- useful_avg_req_len
     |- req_types
     |- sparsity
     |- next-item-train-seq
     |- next-item-valid-seq
     |- next-item-test-seq
     |- data
        |- u_id0
            |- data
            |- train
            |- valid
            |- test
        |- u_id1
        ...
    
    Notes:
    ------
    uid starts from 0
    rid starts from 1(0 for mask)

    '''
    #sort by userid and timestamp
    #for each row:user id, itemid
    #data: (UserID,MovieID)
    data = metadata[np.lexsort((metadata[:, 2], metadata[:, 0])), 0:2];

    uid_set = set(data[:, 0]);
    rid_dict = {};

    udataset = {'req_num' : 0};
    dataset = {};
    uid_idx, rid_idx = 0, 1;
    l_ni_train, l_ni_valid, l_ni_test = [], [], [];
    

    for uid in uid_set:
        su = data[data[:, 0] == uid, 1];
        #delete repeat item
        if len(set(su)) != len(su):
            su = su[np.sort(np.unique(su, return_index = True)[1])];
        #5-core strategy
        if len(su) < 5:
            #This user is not included in the dataset
            continue;
        #reid for req
        for idx in range(len(su)):
            if su[idx] not in rid_dict.keys():
                rid_dict[su[idx]] = rid_idx;
                su[idx] = rid_idx;
                rid_idx += 1;
            else:
                su[idx] = rid_dict[su[idx]];
        #Build the user's dataset
        u_dataset = {};
        #for train
        u_dataset['data'] = torch.from_numpy(su[:-3]).reshape(1, -1);
        #next-item for train
        u_dataset['train'] = torch.tensor(su[-3]).reshape(1, -1);
        l_ni_train.append(u_dataset['train']);
        #next-item for vliation
        u_dataset['valid'] = torch.tensor(su[-2]).reshape(1, -1);
        l_ni_valid.append(u_dataset['valid']);
        #next-item for test
        u_dataset['test'] = torch.tensor(su[-1]).reshape(1, -1);
        l_ni_test.append(u_dataset['test']);

        #Join the total dataset
        dataset[uid_idx] = u_dataset;
        udataset['req_num'] += len(su);
        uid_idx += 1;
    
    udataset['u_num'] = len(dataset);
    udataset['req_types'] = len(rid_dict);
    udataset['data'] = dataset;
    udataset['avg_req_len'] = udataset['req_num']/udataset['u_num'];
    udataset['useful_avg_req_len'] = udataset['avg_req_len'];
    udataset['sparsity'] = cal_sparsity(
        u_num = uid_idx,
        req_types = udataset['req_types'],
        req_num = udataset['req_num']
    )
    udataset['next-item-train-seq'] = torch.as_tensor(l_ni_train).reshape(1, -1);
    udataset['next-item-valid-seq'] = torch.as_tensor(l_ni_valid).reshape(1, -1);
    udataset['next-item-test-seq'] = torch.as_tensor(l_ni_test).reshape(1, -1);
    return udataset


def ml_pcr(src):
    '''Processor for movielens series dataset

    Parameters:
    -----------

    src: chr
    directory of the movielens dataset

    Returns:
    --------

    ml_dataset: dict
    structure
    --- u_num
     |- req_num
     |- req_types
     |- sparsity
     |-next-item-train-seq
     |-next-item-valid-seq
     |-next-item-test-seq
     |- data
        |- u_id0
            |- data
            |- train
            |- valid
            |- test
        |- u_id1
        ...

    Notes:
    ------
    uid starts from 0
    rid starts from 1(0 for mask)
    '''
    #format: UserID::MovieID::Rating::Timestamp
    ml = np.loadtxt(src, dtype = int, usecols = (0, 1, 3));
    return general_pcr(ml);

def rev_pcr(src):
    '''Processor for review dataset

    Parameters:
    -----------

    src: chr
    directory of the review dataset

    Returns:
    --------

    dataset: dict
    structure
    --- u_num
     |- req_num
     |- req_types
     |- sparsity
     |-next-item-train-seq
     |-next-item-valid-seq
     |-next-item-test-seq
     |- data
        |- u_id0
            |- data
            |- train
            |- valid
            |- test
        |- u_id1
        ...

    Notes:
    ------
    uid starts from 0
    rid starts from 1(0 for mask)
    '''
    dict_uid, uid_idx = {}, 0;
    dict_rid, rid_idx = {}, 0;
    data = np.zeros((ju.jsonlen(src), 3), dtype = int);
    metareq_num = 0;

    #Extract data
    for review in ju.jsonparse(src):
        u, r, t= review['reviewerID'], review['asin'], review['unixReviewTime'];
        #Check if the uid exists
        if u in dict_uid.keys() and r in dict_rid.keys():
            #write metadata
            data[metareq_num, :] = np.array([dict_uid[u], dict_rid[r], t]);
        elif u not in dict_uid.keys() and r in dict_rid.keys():
            #occupy an uid
            dict_uid[u] = uid_idx;
            uid_idx += 1;
            #write metadata
            data[metareq_num, :] = np.array([dict_uid[u], dict_rid[r], t]);
        elif u in dict_uid.keys() and r not in dict_rid.keys():
            #occupy an rid
            dict_rid[r] = rid_idx;
            rid_idx += 1
            #write metadata
            data[metareq_num, :] = np.array([dict_uid[u], dict_rid[r], t]);
        else:
            #occupy an uid
            dict_uid[u] = uid_idx;
            uid_idx += 1;
            #occupy an rid
            dict_rid[r] = rid_idx;
            rid_idx += 1
            #write metadata
            data[metareq_num, :] = np.array([dict_uid[u], dict_rid[r], t]);
        metareq_num += 1;
    return general_pcr(data);

def tensor_rid(su, rid_dict, rid_idx, round_):
    '''reid for tensor seq
    '''
    for idx in range(round_):
        if int(su[:, idx]) not in rid_dict.keys():
            rid_dict[int(su[:, idx])] = rid_idx;
            su[:, idx] = rid_idx;
            rid_idx += 1;
        else:
            su[:, idx] = rid_dict[int(su[:, idx])];
    return su, rid_dict, rid_idx;

def cut_and_fill_pcr(dataset, length, fill_mask = 0):
    '''Reprocess the dataset: cut and fill

    Parameters:
    -----------

    dataset:dict

    length:int

    fill_mask:int, optional

    is_aug_mask:bool, optional

    Returns:
    --------

    dataset: dict
    structure
    --- u_num
     |- req_num
     |- req_types
     |- useful_avg_req_len
     |- sparsity
     |-next-item-train-seq
     |-next-item-valid-seq
     |-next-item-test-seq
     |- data
        |- u_id0
            |- data
            |- train
            |- valid
            |- test
        |- u_id1
        ...

    Notes:
    ------
    uid starts from 0
    rid starts from 1(0 for mask)
    '''
    rid_idx, rid_dict = 1, {};
    if fill_mask < dataset['req_types'] and fill_mask > 1:
        raise callback.CustomException(f'ValueError.\nValue of fill_mask [{fill_mask}] not expected to duplicate the serial number {fill_mask}');
    rid_dict[fill_mask] = fill_mask;
    dataset['req_num'] = 0;
    dataset['useful_avg_req_len'] = 0;
    for uid in range(dataset['u_num']):
        #su:(1, req_len)
        su = dataset['data'][uid]['data'];

        if su.shape[1] >= length:
            #cut
            su = su[:, -length:];
            dataset['useful_avg_req_len'] += length;
        else:
            #fill
            dataset['useful_avg_req_len'] += su.shape[1];
            su = torch.cat(
                (
                    torch.ones((1, length - su.shape[1])) * fill_mask,
                    su
                ),
                dim = -1
            );

        #reid for su
        su, rid_dict, rid_idx = tensor_rid(su, rid_dict, rid_idx, su.shape[1]);
        dataset['data'][uid]['data'] = su;
        dataset['req_num'] += su.shape[1];
        #reid for su train
        dataset['data'][uid]['train'], rid_dict, rid_idx = \
            tensor_rid(dataset['data'][uid]['train'], rid_dict, rid_idx, dataset['data'][uid]['train'].shape[1]);
        #reid for su valid
        dataset['data'][uid]['valid'], rid_dict, rid_idx = \
            tensor_rid(dataset['data'][uid]['valid'], rid_dict, rid_idx, dataset['data'][uid]['valid'].shape[1]);
        #reid for su test
        dataset['data'][uid]['test'], rid_dict, rid_idx = \
            tensor_rid(dataset['data'][uid]['test'], rid_dict, rid_idx, dataset['data'][uid]['test'].shape[1]);
    dataset['data'][uid]['data'] = su;
    #reid for su next-item-train-seq
    dataset['next-item-train-seq'], rid_dict, rid_idx = \
        tensor_rid(dataset['next-item-train-seq'], rid_dict, rid_idx, dataset['next-item-train-seq'].shape[1]);
    #reid for su next-item-valid-seq
    dataset['next-item-valid-seq'], rid_dict, rid_idx = \
        tensor_rid(dataset['next-item-valid-seq'], rid_dict, rid_idx, dataset['next-item-valid-seq'].shape[1]);
    #reid for su next-item-test-seq
    dataset['next-item-test-seq'], rid_dict, rid_idx = \
        tensor_rid(dataset['next-item-test-seq'], rid_dict, rid_idx, dataset['next-item-test-seq'].shape[1]);
    dataset['req_types'] = len(rid_dict);
    dataset['avg_req_len'] = dataset['req_num']/dataset['u_num'];
    dataset['useful_avg_req_len'] = dataset['useful_avg_req_len']/dataset['u_num'];
    dataset['sparsity'] = cal_sparsity(
        u_num = dataset['u_num'],
        req_types = dataset['req_types'],
        req_num = dataset['req_num']
    )
    return dataset;


def dataset_divide(data_set, train_ratio = 0.6, valid_ratio = 0.2):
    ''' Divide the dataset

    Parameters:
    -----------

    dataset: dict

    train_ratio: float, optional
        Ratio of training set to data set
        default: 0.6
    
    Returns:
    --------

    train_set: dict

    test_set: dict
    '''
    #divide numbers
    n = data_set['u_num'];
    n_train = int(n * train_ratio);
    n_eval = int(n * valid_ratio);
    #get user id for train and test dataset 
    user_list_train = np.random.choice(n, n_train, replace = False).tolist();
    user_left =  np.array(list(set(range(data_set['u_num'])) - set(user_list_train)));
    user_list_valid = user_left[np.random.choice(user_left.shape[0], n_eval, replace = False)].tolist();
    user_list_test = list(set(range(data_set['u_num'])) - set(user_list_train) - set(user_list_valid));

    #generate train dataset and test dataset
    train_set = {};
    eval_set = {};
    test_set = {};
    train_set['u_num'] = len(user_list_train);
    train_set['req_num'] = data_set['req_num'];
    train_set['req_types'] = data_set['req_types'];
    eval_set['u_num'] = len(user_list_valid);
    eval_set['req_num'] = data_set['req_num'];
    eval_set['req_types'] = data_set['req_types'];
    test_set['u_num'] = len(user_list_test);
    test_set['req_num'] = data_set['req_num'];
    test_set['req_types'] = data_set['req_types'];
    train_data = {};
    eval_data = {};
    test_data = {};
    idx = 0;
    for user_id in user_list_train:
        train_data[idx] = data_set['data'][user_id];
        idx += 1;
    idx = 0;
    for user_id in user_list_test:
        test_data[idx] = data_set['data'][user_id];
        idx += 1;
    idx = 0;
    for user_id in user_list_valid:
        eval_data[idx] = data_set['data'][user_id];
        idx += 1;
    train_set['data'] = train_data;
    eval_set['data'] = eval_data;
    test_set['data'] = test_data;

    return train_set, eval_set, test_set;