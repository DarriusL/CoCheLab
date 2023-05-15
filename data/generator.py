# @Time   : 2023.03.03
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
from data import processor as pcr
import time, torch, torch, os, copy
from lib.callback import CustomException as ce
from lib import glb_var, util

def report(dataset, src):
    '''
    Display information about datasets

    Parameters:
    -----------

    datasets: dict

    src:dataset dir
    '''
    glb_var.get_value('logger').info(
        src + ' \n'
        f'Report\n'
        '==============================\n'
        f'Users: {dataset["u_num"]}\n'
        f'Reqs: {dataset["req_types"]}\n'
        f'Interactions: {dataset["req_num"]}\n'
        f'Avg Req Len: {dataset["avg_req_len"]:.2f}\n'
        f'Avg useful Req Len: {dataset["useful_avg_req_len"]:.2f}\n'
        f'Sparsity: {dataset["sparsity"]:.4f}\n'
        '==============================\n'
        )

def ml_generate(src):
    '''Generate the ml-1m dataset and save it to the target directory

    Parameters:
    -----------

    src: str
    path with filename
    '''
    #process ml-1m
    start = time.time();
    glb_var.get_value('logger').info(f'Processing {src} ...');
    mldataset = pcr.ml_pcr(src = src);
    glb_var.get_value('logger').info(f'Processing {src} complete. time consuming:{util.s2hms(time.time() - start)}\n');
    report(
        dataset = mldataset,
        src = src
    )
    return mldataset;

def rev_generate(src):
    '''Generate a review dataset and save it to the target directory

    Parameters:
    -----------

    src: str
    path with filename

    '''
    #process
    start = time.time();
    glb_var.get_value('logger').info(f'Processing {src} ...');
    dataset = pcr.rev_pcr(src = src);
    glb_var.get_value('logger').info(f'Processing {src} complete. time consuming:{util.s2hms(time.time() - start)}\n');
    report(
        dataset = dataset,
        src = src
    )
    return dataset;

def run_pcr(cfg):
    '''Running function of data processing

    Parameters:
    -----------
    cfg:dict
    Configure for data-processing
    '''
    for dp_cfg in cfg.values():
        save_path, _ = os.path.split(dp_cfg['tgt']);
        if not os.path.exists(save_path):
            os.makedirs(save_path);
        if dp_cfg['data_type'].lower() == 'movielens':
            dataset = ml_generate(dp_cfg['src'])
            torch.save(dataset, dp_cfg['tgt']);
        elif dp_cfg['data_type'].lower() == 'amazon review':
            dataset = rev_generate(dp_cfg['src']);
            torch.save(dataset, dp_cfg['tgt']);
        if dp_cfg['crop_or_fill']:
            dataset_repcr = run_repcr(dataset, dp_cfg['limit_length'], dp_cfg['fill_mask']);
            if dp_cfg['repcr_devide_to_train_valid_test']:
                glb_var.get_value('logger').info(f'divide dataset  ...');
                train, valid, test = pcr.dataset_divide(copy.deepcopy(dataset_repcr), dp_cfg['ratio'][0], dp_cfg['ratio'][1]);
                torch.save({'train': train, 'valid': valid, 'test':test}, dp_cfg['repcr_tgt'])
                glb_var.get_value('logger').info(f'divide dataset  ... complete. \n');
            else:
                torch.save(dataset_repcr, dp_cfg['repcr_tgt']);
    
def run_repcr(dataset, length, fill_mask = 0):
    '''Reporcess
    '''
    start = time.time();
    glb_var.get_value('logger').info(f'Reprocessing, cut or fill to {length}  ...');
    dataset_repcr = pcr.cut_and_fill_pcr(dataset, length, fill_mask);
    glb_var.get_value('logger').info(f'Reprocessing, cut or fill to {length}  ... complete. time consuming:{util.s2hms(time.time() - start)}\n');
    return dataset_repcr

class NextReqDataSet(torch.utils.data.Dataset):
    '''Dataset for the project on next req algorithm

       sample format: 
        su:(batch_size, req_len)
        next-item:(1)

    '''
    def __init__(self, Dataset, mode = 'train') -> None:
        super().__init__();
        #general initial
        self.u_num = Dataset['u_num'];
        self.data = Dataset['data'];
        self.next_item = torch.cat(
            (Dataset['next-item-train-seq'].reshape(1, -1),
             Dataset['next-item-valid-seq'].reshape(1, -1),
             Dataset['next-item-test-seq'].reshape(1, -1)), dim = 0);

        self.valid_data = {};
        self.test_data = {};
        if mode == 'train':
            pass
        elif mode == 'valid':
            for uid in range(self.u_num):
                self.valid_data[uid] = torch.cat(
                    (self.data[uid]['data'],
                     self.next_item[:1, uid:uid + 1]), dim = -1
                )
        elif mode == 'test':
            for uid in range(self.u_num):
                self.test_data[uid] = torch.cat(
                    (self.data[uid]['data'],
                     self.next_item[:1, uid:uid + 1],
                     self.next_item[1:2, uid:uid + 1]), dim = -1
                )
        else:
            glb_var.get_value('logger').error(f'Unrecognized mode [{mode}],only accept the two mode types: train/valid/test.\n')
            raise ce('ModeErorr');
        self.mode = mode;

    def __len__(self):
        return self.u_num;

    def __getitem__(self, index):
        if self.mode == 'train':
            return (index, self.data[index]['data'].squeeze(0).to(torch.int64), self.next_item[0, index].to(torch.int64));
        elif self.mode == 'valid':
            return  index, self.valid_data[index].squeeze(0).to(torch.int64), self.next_item[1, index].to(torch.int64);
        elif self.mode == 'test':
            return index, self.test_data[index].squeeze(0).to(torch.int64), self.next_item[2, index].to(torch.int64);

class GeneralDataSet(torch.utils.data.Dataset):
    def __init__(self, Dataset) -> None:
        super().__init__();
        self.u_num = Dataset['u_num'];
        self.data = {};
        for idx in range(self.u_num):
            self.data[idx] = torch.cat(
                (Dataset['data'][idx]['data'],
                 Dataset['data'][idx]['train'],
                 Dataset['data'][idx]['valid'],
                 Dataset['data'][idx]['test']),
                 dim = -1);

    def __len__(self):
        return self.u_num;

    def __getitem__(self, index):
        return self.data[index].squeeze(0).to(torch.int64);

def get_dataloader(dataset, net_cfg, num_workers, loader_batch, shuffle = False, mode = 'train'):
    if net_cfg['is_cl_method']:
        return torch.utils.data.DataLoader(NextReqDataSet(dataset, mode = mode), 
                                                   num_workers = num_workers,
                                                   pin_memory = True,
                                                   batch_size = loader_batch, 
                                                   shuffle = shuffle);
    elif net_cfg['type'].lower() in ['caser', 'psac_gen']:
        return torch.utils.data.DataLoader(GeneralDataSet(dataset[mode]), 
                                                   num_workers = num_workers,
                                                   pin_memory = True,
                                                   batch_size = loader_batch, 
                                                   shuffle = shuffle);