import torch, time
from lib import glb_var, json_util, util, callback
from Room.officer import Trainer, Tester, get_save_path
from model import *
from data import *

def run_work(config_path, mode = 'train'):
    ''' Run work command

    Parameters:
    -----------
    config_path:str
    path of configure file(json file)

    mode:str, optional
    mode of work: train, test, train_and_test
    default:train
    '''
    #load config
    if mode == 'test':
        config = torch.load(config_path).get('config');
    else:
        config = json_util.jsonload(config_path);
    #set random seed
    torch.manual_seed(config['seed']);
    #initial device 
    if config['train']['gpu_is_available']:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");
    else:
        device = torch.device("cpu");
    glb_var.set_value('device', device);

    #check dateset config
    dataset = torch.load(config['dataset']['path']);
    if config['dataset']['crop_or_fill']:
        t = time.time()
        glb_var.get_value('logger').info(f'Cut and fill dataset to limited length: {config["dataset"]["limit_length"]} ...')
        dataset = run_repcr(dataset, 
                            length = config['dataset']['limit_length'], 
                            fill_mask = config['dataset']['fill_mask']);
        glb_var.get_value('logger').info(f'Processing complete, consuming: {util.s2hms(time.time() - t)}');
        if mode in ['train', 'train_and_test']:
            config['net']['input_types'] = dataset['req_types'];
    
    #get model
    if mode in ['train', 'train_and_test']:
        model = generate_model(config['net']);
        if config['net']['is_net_manual_init']:        
            model.apply(init_weight);
    else:
        model  = torch.load(config_path).get('model');
    
    #conditional update
    if mode in ['train', 'train_and_test']:
        config['train']['model_save_path'] = get_save_path(config);
        glb_var.get_value('logger').info(f"Updata save path:[{config['train']['model_save_path']}]")
        json_util.jsonsave(config, config_path);

    report(config);
    if mode == 'train':
        trainer = Trainer(config, model);
        run_train(trainer, dataset);
    elif mode == 'train_and_test':
        trainer = Trainer(config, model);
        run_train(trainer, dataset);
        if config['test']['gpu_is_available']:
            glb_var.set_value('device', torch.device("cuda:0" if torch.cuda.is_available() else "cpu"));
        else:
            glb_var.set_value('device', torch.device("cpu"));
        tester = Tester(config, torch.load(config['train']['model_save_path']).get('model'));
        run_test(tester, dataset);
    elif mode == 'test':
        if config['test']['gpu_is_available']:
            glb_var.set_value('device', torch.device("cuda:0" if torch.cuda.is_available() else "cpu"));
        else:
            glb_var.set_value('device', torch.device("cpu"));
        tester = Tester(config, model);
        run_test(tester, dataset);
    else:
        glb_var.get_value('logger').error(f'Unrecognized Mode [{mode}], acceptable:(train/test/train_and_test)');
        raise callback.CustomException('ModeError');

def run_train(trainer, dataset):
    '''training the model

    Parameters:
    -----------
    trainer:Room.officer.Trainer
    object to train the model

    dataset:dict
    '''
    t = time.time();
    glb_var.get_value('logger').info('Start training ... ');
    trainer.train(dataset);
    glb_var.get_value('logger').info(f'Training complete, time consuming: {util.s2hms(time.time() - t)}');
    return trainer.model;

def run_test(tester, dataset):
    t = time.time();
    glb_var.get_value('logger').info('Start Testing ... ');
    tester.test(dataset);
    glb_var.get_value('logger').info(f'Testing complete, time consuming: {util.s2hms(time.time() - t)}');

def generate_model(net_cfg_dict):
    '''Generate model base on config

    Parameters:
    -----------
    net_cfg_dict:dict
    configure of model

    Returns:
    --------
    model
    '''
    if net_cfg_dict['type'].lower() == 'cl4srec':
        model = CL4SRec(net_cfg_dict);
    elif net_cfg_dict['type'].lower() == 'duo4srec':
        model = Duo4SRec(net_cfg_dict);
    elif net_cfg_dict['type'].lower() == 'ec4srec':
        model = EC4SRec(net_cfg_dict);
    elif net_cfg_dict['type'].lower() == 'psac_gen':
        model = PSAC_gen(net_cfg_dict);
    else:
        glb_var.get_value('logger').error(f'Unrecognized Mode [{net_cfg_dict["type"].lower()}]');
        raise callback.CustomException('ModelTypeError');
    return model;

def report(config):
    '''print the config
    '''
    glb_var.get_value('logger').info(
        'CacheLab Configure\n'
        '=======================================\n' + json_util.dict2jsonstr(config)
    );

def init_weight(m):
    '''function for initialization
    '''
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.trunc_normal_(m.weight, a=-0.01, b=0.01);
        if m.bias is not None:
            torch.nn.init.trunc_normal_(m.bias, a=-0.01, b=0.01);
    elif isinstance(m, torch.nn.Embedding):
        torch.nn.init.trunc_normal_(m.weight, a=-1, b=1);



