from data import processor as pcr
import time, torch, torch
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

def ml_generate(tgt):
    '''Generate the ml-1m dataset and save it to the target directory

    Parameters:
    -----------

    tgt: str
    path with filename
    '''
    #process ml-1m
    start = time.time();
    src = './data/datasets/meta/ml-1m/ratings.dat';
    glb_var.get_value('logger').info(f'Processing {src} ...');
    mldataset = pcr.ml_pcr(src = src);
    torch.save(mldataset, str(tgt));
    glb_var.get_value('logger').info(f'Processing {src} complete. time consuming:{util.s2hms(time.time() - start)} s\n');
    
    report(
        dataset = mldataset,
        src = src
    )

def rev_generate(src, tgt):
    '''Generate a review dataset and save it to the target directory

    Parameters:
    -----------

    tgt: str
    path with filename

    '''
    #process
    start = time.time();
    glb_var.get_value('logger').info(f'Processing {src} ...');
    dataset = pcr.rev_pcr(src = src);
    torch.save(dataset, str(tgt));
    glb_var.get_value('logger').info(f'Processing {src} complete. time consuming:{util.s2hms(time.time() - start)} s\n');
    report(
        dataset = dataset,
        src = src
    )

def lite_generate(tgt):
    '''
    Generate a lite dataset and save it to the target path

    Parameters:

    tgt: str
    directory to save the file
    '''

    #process ml-1m
    ml_generate(str(tgt) + '/ml.data');
    #process appliances
    rev_generate(
        src = './data/datasets/meta/appliances/Appliances_lite.json',
        tgt = str(tgt) + '/Appliances_lite.data'
        )
    #process beauty
    rev_generate(
        src = './data/datasets/meta/beauty/All_Beauty_lite.json',
        tgt = str(tgt) + '/All_Beauty_lite.data'
    )
    #process music
    rev_generate(
        src = './data/datasets/meta/music/Digital_Music_lite.json',
        tgt = str(tgt) + '/Digital_Music_lite.data'
    )
    #process kindle
    rev_generate(
        src = './data/datasets/meta/kindle/Kindle_Store_lite.json',
        tgt = str(tgt) + '/Kindle_Store_lite.data'
    )

def complete_generate(tgt):
    '''
    Generate a complete dataset and save it to the target path

    Parameters:

    tgt: str
    directory to save the file
    '''
    #process ml-1m
    ml_generate(str(tgt) + '/ml.data');
    #process appliances
    rev_generate(
        src = './data/datasets/meta/appliances/Appliances.json',
        tgt = str(tgt) + '/Appliances.data'
        )
    #process beauty
    rev_generate(
        src = './data/datasets/meta/beauty/All_Beauty.json',
        tgt = str(tgt) + '/All_Beauty.data'
    )
    #process music
    rev_generate(
        src = './data/datasets/meta/music/Digital_Music.json',
        tgt = str(tgt) + '/Digital_Music.data'
    )
    #process kindle
    rev_generate(
        src = './data/datasets/meta/kindle/Kindle_Store.json',
        tgt = str(tgt) + '/Kindle_Store.data'
    )

def run_pcr(cfg):
    '''Running function of data processing

    Parameters:
    -----------
    cfg:dict
    Configure for data-processing
    '''
    if cfg['type'] == 'lite':
        if cfg['tgt'] is None:
            tgt = './data/datasets/process/lite';
        lite_generate(tgt = tgt);
    elif cfg['type'] == 'complete':
        if cfg['tgt'] is None:
            tgt = './data/datasets/process/complete';
        complete_generate(tgt = tgt);
    else:
        glb_var.get_value('logger').error(f'Unrecognized type[{cfg["type"]}],only two types (lite and complete) accepted.\n')
        raise ce('TypeError')
    
def run_repcr(dataset, length, fill_mask = 0):
    '''Reporcess
    '''
    dataset = pcr.cut_and_fill_pcr(dataset, length, fill_mask);
    report(dataset, 'reprocess dataset');
    return dataset

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