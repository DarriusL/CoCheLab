# @Time   : 2023.03.03
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com

import torch, time,os
from lib import glb_var, json_util, util, callback
from Room.officer import Trainer, Tester, ConventionalTester, get_save_path
from model import *
from data import *

logger = glb_var.get_value('logger');

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
    lab_cfg = json_util.jsonload('./config/lab_cfg.json');
    if mode == 'test':
        config = json_util.jsonload(config_path);
        _, config['train']['model_save_path'] = os.path.split(config['train']['model_save_path']);
        cfg_root, _ = os.path.split(config_path);
        config['train']['model_save_path'] = cfg_root + '/' + config['train']['model_save_path'];
        logger.info(f"Updata save path:[{config['train']['model_save_path']}]");
        json_util.jsonsave(config, config_path);
        del cfg_root;
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
    #set training constant
    if config['train']['use_amp']:
        glb_var.set_value('mask_to_value', lab_cfg['constant']['use_amp_true']['mask_to_value']);
        glb_var.set_value('eps', lab_cfg['constant']['use_amp_true']['eps']);
    else:
        glb_var.set_value('mask_to_value', lab_cfg['constant']['use_amp_false']['mask_to_value']);
        glb_var.set_value('eps', lab_cfg['constant']['use_amp_false']['eps']);

    #check dateset config
    dataset = torch.load(config['dataset']['path']);
    if config['dataset']['crop_or_fill']:
        t = time.time()
        logger.info(f'Cut and fill dataset to limited length: {config["dataset"]["limit_length"]} ...')
        dataset = run_repcr(dataset, 
                            length = config['dataset']['limit_length'], 
                            fill_mask = config['dataset']['fill_mask']);
        logger.info(f'Processing complete, consuming: {util.s2hms(time.time() - t)}');
        if mode in ['train', 'train_and_test']:
            config['net']['input_types'] = dataset['req_types'];
    elif mode in ['train', 'train_and_test']:
        #Using a processed dataset
        config['net']['input_types'] = dataset['train']['req_types'];

    
    #get model
    if mode in ['train', 'train_and_test']:
        model = generate_model(config);
    else:
        model  = torch.load(config['train']['model_save_path']);
    
    #conditional update
    if mode in ['train', 'train_and_test']:
        config['train']['model_save_path'] = get_save_path(config);
        logger.info(f"Updata save path:[{config['train']['model_save_path']}]")
        json_util.jsonsave(config, config_path);
    elif mode == 'test' and config['net']['type'].lower() in ['fifo', 'lru', 'lfu']:
        config['test']['model_save_path'] = get_save_path(config);
        logger.info(f"Updata save path:[{config['test']['model_save_path']}]")
        json_util.jsonsave(config, config_path);

    report(config, lab_cfg);
    result = None;
    if mode == 'train':
        if config['net']['type'].lower() in ['fifo', 'lru', 'lfu']:
            logger.error(f"{config['net']['type']} is not supported using {mode} mode");
            raise RuntimeError;
        trainer = Trainer(config, model);
        run_train(trainer, dataset);
    elif mode == 'train_and_test':
        if config['net']['type'].lower() in ['fifo', 'lru', 'lfu']:
            logger.error(f"{config['net']['type']} is not supported using {mode} mode");
            raise RuntimeError;
        trainer = Trainer(config, model);
        run_train(trainer, dataset);
        if config['test']['gpu_is_available']:
            glb_var.set_value('device', torch.device("cuda:0" if torch.cuda.is_available() else "cpu"));
        else:
            glb_var.set_value('device', torch.device("cpu"));
        tester = Tester(config, torch.load(config['train']['model_save_path']));
        result = run_test(tester, dataset);
    elif mode == 'test':
        if config['test']['gpu_is_available']:
            glb_var.set_value('device', torch.device("cuda:0" if torch.cuda.is_available() else "cpu"));
        else:
            glb_var.set_value('device', torch.device("cpu"));
            if config['net']['type'].lower() in ['fifo', 'lru', 'lfu']:
                tester = ConventionalTester(config, model);
            else:
                tester = Tester(config, model);
        result = run_test(tester, dataset);
    else:
        logger.error(f'Unrecognized Mode [{mode}], acceptable:(train/test/train_and_test)');
        raise callback.CustomException('ModeError');

    if config['email_reminder']:
        subject = 'CacheLab Reminder';
        if result is not None:
            str_add = 'The test result is showed below:\n' + result;
        else:
            str_add = '';
        content = 'CacheLab user:\n\nThe lab operation is over, please return to view the results as soon as possible.' + str_add + \
            '\n\nCacheLab\n'+util.get_date(' ');
        email_cfg = lab_cfg['email_reminder'];
        callback.send_smtp_emil(email_cfg['sender'], email_cfg['receiver'], email_cfg['password'],
                                    subject, content, email_cfg['port'], email_cfg['sever']);


def run_train(trainer, dataset):
    '''train the model

    Parameters:
    -----------
    trainer:Room.officer.Trainer
    object to train the model

    dataset:dict
    '''
    t = time.time();
    logger.info('Start training ... ');
    trainer.train(dataset);
    logger.info(f'Training complete, time consuming: {util.s2hms(time.time() - t)}');
    return trainer.model;

def run_test(tester, dataset):
    '''test the trained model
    '''
    t = time.time();
    logger.info('Start Testing ... ');
    result = tester.test(dataset);
    logger.info(f'Testing complete, time consuming: {util.s2hms(time.time() - t)}');
    return result;

def generate_model(cfg):
    '''Generate model base on config

    Parameters:
    -----------
    net_cfg_dict:dict
    configure of model

    Returns:
    --------
    model
    '''
    net_cfg_dict = cfg['net'];
    if net_cfg_dict['type'].lower() == 'cl4srec':
        model = CL4SRec(net_cfg_dict);
    elif net_cfg_dict['type'].lower() == 'duo4srec':
        model = Duo4SRec(net_cfg_dict);
    elif net_cfg_dict['type'].lower() == 'ec4srec':
        model = EC4SRec(net_cfg_dict);
    elif net_cfg_dict['type'].lower() == 'psac_gen':
        model = PSAC_gen(net_cfg_dict, cfg['train']['batch_size']);
    elif net_cfg_dict['type'].lower() == 'caser':
        model = Caser(net_cfg_dict, cfg['train']['batch_size']);
    elif net_cfg_dict['type'].lower() == 'egpc':
        model = EGPC(net_cfg_dict);
    elif net_cfg_dict['type'].lower() == 'fifo':
        model = FIFO(net_cfg_dict);
    elif net_cfg_dict['type'].lower() == 'lru':
        model = LRU(net_cfg_dict);
    elif net_cfg_dict['type'].lower() == 'lfu':
        model = LFU(net_cfg_dict);
    else:
        logger.error(f'Unrecognized Mode [{net_cfg_dict["type"].lower()}]');
        raise callback.CustomException('ModelTypeError');
    return model;

def report(config, lab_cfg):
    '''print the info and check config
    '''
    #check config
    keys = ['net', 'dataset', 'linux_fast_num_workers', 'email_reminder', 'seed', 'train', 'test'];
    train_keys = ['batch_size', 'max_epoch', 'valid_step', 'stop_train_step_valid_not_improve', 'gpu_is_available', 'use_amp', 
                  'optimizer_type', 'learning_rate', 'weight_decay', 'betas', 'use_lr_schedule', 'lr_max', 'metric_less',
                  'save', 'model_save_path', 'end_save'];
    cfg_keys = config.keys();
    for key in keys:
        if key not in cfg_keys:
            logger.error(f'Config miss key [{key}]');
            raise callback.CustomException('ConfigError');
        elif key == 'train':
            cfg_train_keys = config['train'].keys();
            for subkey in train_keys:
                if subkey not in cfg_train_keys:
                    logger.error(f'Config key [train] miss subkey [{subkey}]');
                    raise callback.CustomException('ConfigError');

    #buffer size
    if 'posenc_buffer_size' in config['net'].keys():
        if config['dataset']['limit_length'] + 3 > config['net']['posenc_buffer_size']:
            logger.error('Parameter settings on net [posenc_buffer_size] and on dataset [limit_length] conflict');
            raise callback.CustomException('ConfigError');
    #net config
    if config['net']['type'].lower() in ['cl4srec', 'ec4srec', 'duo4srec', 'egpc'] and not config['net']['is_cl_method']:
            logger.error(f'Net type [{config["net"]["type"]}] is Contrastive Learning, please set [is_cl_method] to [True]');
            raise callback.CustomException('ConfigError');
    if config['email_reminder']:
        try:
            smtp = callback.smtplib.SMTP();
            smtp.connect(lab_cfg['email_reminder']['sever'], lab_cfg['email_reminder']['port']);
            smtp.login(lab_cfg['email_reminder']['sender'], lab_cfg['email_reminder']['password']);
            smtp.close();
        except:
            logger.error('Please enter the config directory to configure the lab_cfg.json file');
            raise callback.CustomException('ConfigError');
    logger.info(
    f'CacheLab Configuration report:\n'
    '------------------------------------\n'
    f'Constant settings:\nDevice: [{glb_var.get_value("device")}]\n'
    f'Eps: [{glb_var.get_value("eps")}]\nMask_to_value: [{glb_var.get_value("mask_to_value")}]\n'
    '------------------------------------\n'
    );




