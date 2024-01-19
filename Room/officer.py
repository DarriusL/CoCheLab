# @Time   : 2023.03.03
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com

import torch, os, time, platform, copy
from lib import util, glb_var, callback, json_util
import numpy as np
from data import augmentation, generator
import matplotlib.pyplot as plt
from collections import Counter

logger = glb_var.get_value('logger');

def get_save_path(cfg):
    '''Gnerate save path
    '''
    if cfg['net']['type'].lower() in ['caser','psac_gen']:
        return './data/saved/' + cfg['net']['type'].lower() + '/' + cfg['dataset']['type'] + '/' + f'{cfg["net"]["d"]}_{cfg["net"]["n_kernels"]}/model.model';
    elif cfg['net']['type'].lower() in ['fifo','lru','lfu']:
        return './data/saved/' + cfg['net']['type'].lower() + '/' + cfg['dataset']['type'] + '/model.model';
    elif cfg['net']['is_norm_first']:
        norm_type = 'pre';
    else:
        norm_type = 'post';

    path = './data/saved/' + \
            cfg['net']['type'].lower() + '/' + cfg['dataset']['type'] + '/' + norm_type + \
            f'_{cfg["net"]["d"]}_{cfg["net"]["d_fc"]}_{cfg["net"]["n_heads"]}_{cfg["net"]["n_layers"]}/model.model';

    return path;

class AbstractTrainer():
    '''Abstract parent trainer class
    '''
    def __init__(self, train_cfg_dict, model) -> None:
        util.set_attr(self, train_cfg_dict);
        self.train_loss = [];
        self.valid_loss = [];
        self.valid_min_loss = np.inf if self.metric_less else -np.inf;
        self.device = glb_var.get_value('device');
        if self.use_amp:
            self.scaler = torch.cuda.amp.grad_scaler.GradScaler();
        self.model = model.to(self.device);
        #optimizer
        if self.optimizer_type.lower() == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate, betas = self.betas, weight_decay = self.weight_decay);
        elif self.optimizer_type.lower() == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = self.learning_rate, betas = self.betas, weight_decay = self.weight_decay);
        else:
            logger.warning(f"Unrecognized optimizer[{self.optimizer_type.lower()}], set default Adam optimizer");
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate);

        #lr decay
        if self.use_lr_schedule:
            self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer = self.optimizer,
                total_steps = self.max_epoch,
                max_lr = self.lr_max
            )
        if self.save:
            self.save_path, _ = os.path.split(self.model_save_path);
        else:
            self.save_path = './cache/unsaved_data/[' + util.get_date('_') + ']';
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
    
    def train(self, train_data):
        logger.error('Method needs to be called after being implemented');
        raise callback.CustomException('NotImplementedError');

    def _check_nan(self, loss):
        if torch.isnan(loss):
            logger.error('Loss is nan.\nHint:\n(1)Check the loss function;\n'
                              '(2)Checks if the constant used is in the range between [-6.55 x 10^4]~[6.55 x 10^4]\n'
                              '(3)Not applicable for automatic mixed-precision acceleration.');
            raise callback.CustomException('ValueError');

    def _save(self):
        '''Save the model and configure
        '''
        torch.save(self.model, self.save_path + '/model.model');
        json_util.jsonsave(self.cfg, self.save_path + '/config.json');
        logger.info(f'Save path: {self.save_path}')

class AbstractTester():
    '''Abstract parent trainer class
    '''
    def __init__(self, test_cfg_dict, model) -> None:
        util.set_attr(self, test_cfg_dict);
        self.device = glb_var.get_value('device');
        if model.type.lower() not in ['fifo', 'lru', 'lfu']:
            self.model = model.to(self.device);
        else:
            self.model = model;
        

    def test(self, test_data):
        logger.error('Method needs to be called after being implemented');
        raise callback.CustomException('NotImplementedError');

class Trainer(AbstractTrainer):
    '''Trainer for general algorithm

    Parameters:
    -----------
    config:dict

    model:torch.nn.Module
    '''
    def __init__(self, config, model) -> None:
        super().__init__(config['train'],  model)
        self.cfg = config;
        if self.model.is_cl_method:
            #augmentation
            self.aug = augmentation.get_augmentation(copy.deepcopy(config['augmentation']));

    def _run_epoch(self, data, mode = 'train'):
        ''' epcoh for trian and validation
        Parameters:
        ----------

        data:
        .(1)tuple, when is cl method
            (index, su_batch, next_req_bacth)
            >su_batch:(batch_size, seq_len)
            >next_req_bacth:(batch_size)
        .(2)torch.Tensor, when is caser-based
            (batch_size, req_len)

        mode:str,optional
        'train': backward
        'valid':no_grad

        Returns:
        --------
        loss
        '''

        if self.model.is_cl_method:
            index, su_batch, next_req_bacth = data;
            su_batch, next_req_bacth = su_batch.to(self.device), next_req_bacth.to(self.device);
        elif self.model.type.lower() in ['caser', 'psac_gen']:
            data = data.to(self.device).unfold(-1, self.model.L + 1, self.model.L);
            su_batch = data[:, :, :self.model.L];
            next_req_bacth = data[:, :, -1]


        epoch_loss = [];
        if self.model.is_cl_method:
            #augmentation
            if self.model.type.lower() == 'duo4srec':
                self.aug.update_scale([next_req_bacth])
            elif self.model.type.lower() in ['ec4srec', 'egpc'] and 'retrieval' in self.cfg['augmentation']['operator']:
                self.aug.scale[-1] = next_req_bacth;
            self.aug.sample_opr();

            if self.model.type.lower() in ['ec4srec', 'egpc']:
                #su_batch_operat_list:tuple
                if mode == 'train':
                    su_batch_operat_list = self.aug.operate(self.impt_score_train[index, :].clone(), su_batch.clone());
                elif mode == 'valid':
                    su_batch_operat_list = self.aug.operate(self.impt_score_valid[index, :].clone(), su_batch.clone());
                else:
                    logger.error('Unsupported mode type');
                    raise callback.CustomException('ModeError');
            else:
                #su_batch_operat_list:list
                su_batch_operat_list = self.aug.operate(su_batch.clone());

            for batch_idx in range(self.batch_size):
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache();
                #calculate loss
                if self.use_amp and mode == 'train':
                    self.optimizer.zero_grad();
                    #use amp when training
                    with torch.cuda.amp.autocast_mode.autocast():
                        loss = self.model.cal_loss(su_batch, next_req_bacth, su_batch_operat_list, batch_idx);
                    self._check_nan(loss);
                    self.scaler.scale(loss).backward();
                    torch.nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), max_norm=1);
                    self.scaler.step(self.optimizer);
                    self.scaler.update();
                elif mode == 'train':
                    self.optimizer.zero_grad();
                    #not using amp when training
                    loss = self.model.cal_loss(su_batch, next_req_bacth, su_batch_operat_list, batch_idx);
                    self._check_nan(loss);
                    loss.backward();
                    self.optimizer.step();
                else:
                    #validation
                    loss = self.model.cal_loss(su_batch, next_req_bacth, su_batch_operat_list, batch_idx);
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache();
                epoch_loss.append(loss.item());
        else:
            #calculate loss
            if self.use_amp and mode == 'train':
                self.optimizer.zero_grad();
                #use amp when training
                with torch.cuda.amp.autocast_mode.autocast():
                    loss = self.model.cal_loss(su_batch, next_req_bacth);
                self._check_nan(loss);
                self.scaler.scale(loss).backward();
                torch.nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), max_norm=1);
                self.scaler.step(self.optimizer);
                self.scaler.update();
            elif mode == 'train':
                self.optimizer.zero_grad();
                #not using amp when training
                loss = self.model.cal_loss(su_batch, next_req_bacth);
                self._check_nan(loss);
                loss.backward();
                self.optimizer.step();
            else:
                #validation
                loss = self.model.cal_loss(su_batch, next_req_bacth);
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache();
            epoch_loss.append(loss.item());
        return np.mean(epoch_loss)

    def _train_epoch(self, train_data):
        '''Train the model in one epoch
        
        Parameters:
        ----------

        train_data:
        .(1)tuple, when is cl method
            (index, su_batch, next_req_bacth)
            >su_batch:(batch_size, seq_len)
            >next_req_bacth:(batch_size)
        .(2)torch.Tensor, when is caser-based
            (batch_size, req_len)

        Returns:
        --------
        loss
        '''
        #train mode
        self.model.train();
        return self._run_epoch(train_data, mode = 'train');

    @torch.no_grad()
    def _valid_epoch(self, valid_data):
        '''valid the model in one epoch
        
        Parameters:
        ----------

        valid_data:
        .(1)tuple, when is cl method
            (index, su_batch, next_req_bacth)
            >su_batch:(batch_size, seq_len)
            >next_req_bacth:(batch_size)
        .(2)torch.Tensor, when is caser-based
            (batch_size, req_len)

        Returns:
        --------
        loss
        '''
        #eval mode
        self.model.eval();
        epoch_loss = self._run_epoch(valid_data, mode = 'valid');
        return epoch_loss;

    def train(self, dataset):
        '''run the training and save related data

        Parameters:
        -----------
        dataset
        '''
        valid_not_improve_cnt = 0;
        if platform.system().lower() == 'linux':
            num_workers = self.cfg['linux_fast_num_workers'];
        else:
            num_workers = 0;
        #initial important score
        if self.cfg['net']['type'].lower() in ['ec4srec', 'egpc']:
            self.impt_score_train = (torch.ones((dataset['u_num'], self.cfg['dataset']['limit_length'])) / \
                self.cfg['dataset']['limit_length']).to(self.device);
            self.impt_score_valid = (torch.ones((dataset['u_num'], self.cfg['dataset']['limit_length'] + 1)) / \
                self.cfg['dataset']['limit_length']).to(self.device);
        loader_batch = self.batch_size;
        train_loader = generator.get_dataloader(dataset, self.cfg['net'], num_workers, loader_batch, shuffle = True, mode = 'train');
        valid_loader = generator.get_dataloader(dataset, self.cfg['net'], num_workers, loader_batch, shuffle = True, mode = 'valid');
        t = time.time();
        for epoch in range(self.max_epoch):
            train_data = iter(train_loader).__next__();
            #train
            self.train_loss.append(self._train_epoch(train_data));
            logger.info(f'[{self.model.type}]-[train]\n'
                            f'[epoch: {epoch + 1}/{self.max_epoch}] - train loss:{self.train_loss[-1]:.8f} - '
                            f'lr:{self.optimizer.param_groups[0]["lr"]}\n'
                            f'Accumulated training time:[{util.s2hms(time.time() - t)}] - '
                            f'Estimated time remaining:[{util.s2hms((time.time() - t)/(epoch + 1) * (self.max_epoch - epoch - 1))}]');
            if self.use_lr_schedule:
                self.lr_scheduler.step();
            if (epoch + 1)%self.valid_step == 0:
                valid_data = iter(valid_loader).__next__();
                with torch.no_grad():
                    if self.model.is_cl_method:
                        req_types = len(Counter(self.model(valid_data[1].clone().to(self.device)).argmax(dim = -1).tolist()));
                    elif self.model.type.lower() in ['caser', 'psac_gen']:
                        req_types = len(Counter(self.model(valid_data.unfold(-1, self.model.L + 1, self.model.L)[:, :, :self.model.L]
                                                           .clone().to(self.device)).argmax(dim = -1).reshape(-1).tolist()));
                #valid
                self.valid_loss.append(self._valid_epoch(valid_data));
                if self.metric_less:
                    #less, better
                    if self.valid_loss[-1] < self.valid_min_loss:
                        self._save();
                        self.valid_min_loss = self.valid_loss[-1];
                        valid_not_improve_cnt = 0;
                    else:
                        valid_not_improve_cnt += 1;
                else:
                    #bigger, better
                    if self.valid_loss[-1] > self.valid_min_loss:
                        self._save()
                        self.valid_min_loss = self.valid_loss[-1];
                        valid_not_improve_cnt = 0;
                    else:
                        valid_not_improve_cnt += 1;

                logger.info(f'[{self.model.type}]-[valid]\n'
                                 f'[epoch: {epoch + 1}/{self.max_epoch}]- valid loss:{self.valid_loss[-1]:.8f} - '
                                 f'valid min loss: {self.valid_min_loss:.8f} - req_types:{req_types} - '
                                 f'valid_not_improve_cnt:{valid_not_improve_cnt}');
                if valid_not_improve_cnt >= self.stop_train_step_valid_not_improve:
                    logger.info('Meet the set requirements, stop training');
                    break;
            if self.cfg['net']['type'].lower() in ['ec4srec', 'egpc'] and epoch + 1 != self.max_epoch:
                #if it's EC4SRec and meet the update imptscore strp
                if (epoch + 1)%self.model.impt_score_step == 0:
                    self.impt_score_train, self.impt_score_valid = self.model.cal_impt_score(dataset);
        if self.end_save:
            torch.save(self.model, self.save_path + '/endmodel.model');
            
        plt.figure(figsize = (10, 6));
        plt.plot(np.arange(0, len(self.train_loss)) + 1, self.train_loss, label = 'train loss');
        plt.plot(np.arange(self.valid_step - 1, len(self.train_loss), self.valid_step) + 1, self.valid_loss, label = 'valid loss');
        plt.xlabel('epoch');
        plt.ylabel('loss');
        plt.yscale('log');
        plt.legend(loc='upper right')
        plt.savefig(self.save_path + '/loss.png', dpi = 400);

class Tester(AbstractTester):
    '''Tester for general algorithm 

    Parameters:
    -----------
    config:dict

    model:torch.nn.Module
    '''
    def __init__(self, config, model) -> None:
        super().__init__(config['test'], model);
        self.model.eval();
        self.cfg = config;
        if self.save:
            self.save_path, _ = os.path.split(config['train']['model_save_path']);
    
    def _cal_hitrate_and_ndcg_atk(self, su_batch, next_req_batch_pre, next_req_batch, at_k):
        '''Calculate Hitrate@k and NDCG@k

        Parameters:
        -----------
        su_batch:torch.Tensor
        (batch_size, req_len)

        next_req_batch_pre:torch.Tensor
        (batch_size)

        next_req_batch:torch.Tensor
        (batch_size)

        at_k:list

        Returns:
        --------
        hr:torch.Tensor
        (len(at_k))

        ndcg:torch.Tensor
        (len(at_k))
        '''
        batch_size = su_batch.shape[0];
        #Representation of prediction requests:
        if self.model.is_cl_method:
            #(batch_size, d)
            h_batch_pre = self.model.encoder(next_req_batch_pre.unsqueeze(-1));
        elif self.model.type.lower() in ['caser', 'psac_gen']:
            #(batch_size, slide_len, d)
            h_batch_pre = self.model.encoder(next_req_batch_pre);
        hr = torch.zeros((len(at_k))).to(self.device);
        ndcg = torch.zeros_like(hr).to(self.device);
        for batch_idx in range(batch_size):
            #uninteract:(uninteract_num)
            uninteract = torch.as_tensor(list(set(torch.arange(self.model.input_types).tolist()) - \
                set(su_batch[batch_idx, :].reshape(-1).tolist()))).to(self.device);
            #Representation of cache_reqs:(100, d)
            if self.model.is_cl_method:
                #cache_reqs:(100, 1)
                cach_reqs = torch.cat(
                    (next_req_batch[[batch_idx]], 
                    uninteract[np.random.choice(uninteract.shape[0], 99, replace = False)]),
                    dim = 0
                ).unsqueeze(-1);
                h_cache_reqs = self.model.encoder(cach_reqs);
                #scores:(100):Ascending
                scores = torch.matmul(h_cache_reqs, h_batch_pre[[batch_idx], :].transpose(0, 1)).squeeze(-1);
            elif self.model.type.lower() in ['caser', 'psac_gen']:
                #cache_reqs:(100, 1)
                cach_reqs = torch.cat(
                    (next_req_batch[[batch_idx], -1], 
                    uninteract[np.random.choice(uninteract.shape[0], 99, replace = False)]),
                    dim = 0
                ).unsqueeze(-1);
                h_cache_reqs = self.model.encoder(cach_reqs.reshape(-1, 1, 1)).reshape(-1, self.model.d);
                #scores:(100):Ascending
                scores = torch.matmul(h_cache_reqs, h_batch_pre[[batch_idx], -1, :].transpose(0, 1)).squeeze(-1);
            rank = torch.arange(scores.shape[0]);
            rank[torch.argsort(scores, stable=True, descending=True)] = rank.clone();
            #count hit number
            hr[rank[0] < torch.as_tensor(at_k)] += 1;
            #add ndcg
            ndcg[rank[0] < torch.as_tensor(at_k)] += 1/np.log2(rank[0] + 2);
        hr = hr/batch_size;
        ndcg = ndcg/batch_size;
        return hr, ndcg;
            
    def _slide_item_cache(self, data, alter_dict):
        '''

        Parameters:
        -----------
        data:torch.Tensor
        (batch_size, req_len)

        alter_dict:dict
        '''
        #(alter_topk)
        if self.model.is_cl_method:
            for batch_id in range(self.batch_size):
                #su:(slide_len, T)
                su = data[batch_id, :].unfold(-1, self.slide_T + 1, self.slide_T)[:, :self.slide_T];
                for j in range(su.shape[0]):
                    _, logits_topk = self.model(su[:j+1, :].reshape(1, -1)).topk(self.alter_topk, dim = -1);
                    alter_topk = logits_topk.squeeze(0).tolist();
                    for i in range(self.alter_topk):
                        if alter_topk[i] in alter_dict:
                            alter_dict[alter_topk[i]] += 1;
                        else:
                            alter_dict[alter_topk[i]] = 1;
        elif self.model.type.lower() in ['caser', 'psac_gen']:
            #su_batch:(batch_size, slide_len, L)
            su_batch = data.unfold(-1, self.slide_T + 1, self.slide_T)[:, :, :self.slide_T];
            _, logits_topk = self.model(su_batch).topk(self.alter_topk, dim = -1);
            for batch_id in range(self.batch_size):
                for i in range(su_batch.shape[1]):
                    alter_topk = logits_topk[batch_id, i, :].tolist();
                    for i in range(self.alter_topk):
                        if alter_topk[i] in alter_dict:
                            alter_dict[alter_topk[i]] += 1;
                        else:
                            alter_dict[alter_topk[i]] = 1;
        
    
    def _caching_and_cal_qoe_trafficload(self, data, cache_size_list):
        '''Calculate QoE and Traffice Load at cache_size

        Parameters:
        -----------
        data:torch.Tensor
        (batch_size, req_len)

        cache_size_list:list

        Returns:
        --------
        QoE:dict

        TrafficLoad:dict
        '''
        batch_size, req_len = data.shape;
        #Get alternative cache dict
        alter_dict = {};
        self._slide_item_cache(data, alter_dict);
        QoE = {};
        TrafficLoad = {};
        for cache_size in cache_size_list:
            cache_num = int(np.round(self.bs_storagy * cache_size));
            #Get the cache corresponding to cache size
            if len(alter_dict) <= cache_num:
                cache_set = set(alter_dict.keys());
            else:
                #choose top-cache_num
                cache_set = set(torch.as_tensor(list(alter_dict.keys()))
                            [torch.argsort(torch.as_tensor(list(alter_dict.values())), descending=True)[:cache_num]].tolist());
            logger.debug('Tester._caching_and_cal_qoe_trafficload\n'
                            f'cache_num: {cache_num} - len(cache_set): {len(cache_set)}')
            #calculate qoe and trafficload
            qoe, userload, allload = 0, 0, 0;
            for batch_id in range(batch_size):
                #R:real data set
                R = set(data[batch_id, :].unfold(-1, self.slide_T + 1, self.slide_T)[:, -1].tolist());
                if len(cache_set & R) > (req_len - data[batch_id, :].eq(0).sum().item())*self.cache_satisfaction_ratio:
                    qoe += 1;
                userload += len(R - cache_set);
                allload += len(R);
            QoE[cache_size] = qoe/batch_size;
            TrafficLoad[cache_size] = userload/allload;
        return QoE, TrafficLoad;

    @torch.no_grad()
    def test(self, dataset):
        '''Test the trained model

        Parameters:
        -----------
        dataset:dict
        Processed dataset
        '''
        #Adjust the number of workers of dataloader according to the system CPU and system
        if platform.system().lower() == 'linux':
            num_workers = self.cfg['linux_fast_num_workers'];
        else:
            num_workers = 0;
        result = {'HitRate':{}, 'NDCG':{}, 'QoE':{}, 'TrafficLoad':{}};
        #initial
        for cs in self.cache_size:
            result['QoE'][cs] = 0;
            result['TrafficLoad'][cs] = 0;
        test_loader = generator.get_dataloader(dataset, self.cfg['net'], num_workers, self.batch_size, shuffle = True, mode = 'test');
        if self.model.is_cl_method:
            n_step = int(np.ceil(dataset['u_num']/self.batch_size)) * 2;
        elif self.model.type.lower() in ['caser', 'psac_gen']:
            n_step = int(np.ceil(dataset['test']['u_num']/self.batch_size)) * 2;
        HitRate, NDCG= torch.zeros(len(self.metrics_at_k)).to(self.device), torch.zeros(len(self.metrics_at_k)).to(self.device);
        for i in range(n_step):
            t_reason = 0;
            t = time.time();
            qoe_batch, tl_batch = [], [];
            #test_data:(batch_size, seq_len)
            #next_req:(batch_size, 1)
            if self.model.is_cl_method:
                _, test_data, next_req = iter(test_loader).__next__();
                test_data, next_req = test_data.to(self.device), next_req.to(self.device);
                #next_req_logits:(batch_size, req_types)
                with torch.no_grad():
                    t_r = time.time();
                    next_req_logits = self.model(test_data);
                    t_reason += (time.time() - t_r);
                #next_req:(batch_size)
                next_req_pre = next_req_logits.argmax(dim = -1);
                logger.debug(f'pre_types:{len(Counter(next_req_pre.tolist()))}');
                #new data:(batch_size, seq_len)
                data = torch.cat((test_data, next_req.unsqueeze(-1)), dim = -1);
            elif self.model.type.lower() in ['caser', 'psac_gen']:
                data = iter(test_loader).__next__();
                #(batch_size, req_len)
                data = data.to(self.device);
                data_slide = data.unfold(-1, self.model.L + 1, self.model.L);
                #
                test_data = data_slide[:, :, :self.model.L];
                next_req = data_slide[:, :, -1];
                with torch.no_grad():
                    #(batch_size, slide_len, req_types)
                    t_r = time.time();
                    next_req_logits = self.model(test_data);
                    t_reason += (time.time() - t_r);
                next_req_pre = next_req_logits.argmax(dim = -1);
                logger.debug(f'pre_types:{len(Counter(next_req_pre.reshape(-1).tolist()))}');

            #calculate hit rate and ndcg
            hitrate, ndcg = self._cal_hitrate_and_ndcg_atk(test_data, next_req_pre, next_req, self.metrics_at_k);
            HitRate += hitrate;
            NDCG += ndcg;
            #calculate qoe and traffic load
            qoe, trafficload = self._caching_and_cal_qoe_trafficload(data, self.cache_size);
            for cs in self.cache_size:
                result['QoE'][cs] += qoe[cs];
                qoe_batch.append(qoe[cs]);
                result['TrafficLoad'][cs] += trafficload[cs];
                tl_batch.append(trafficload[cs]);
            self._report_batch_result(t, i, n_step, hitrate, ndcg, qoe_batch, tl_batch);
        HitRate /= n_step;
        NDCG /= n_step;
        for cs in self.cache_size:
            result['QoE'][cs] /= n_step;
            result['TrafficLoad'][cs] /= n_step;
        result['HitRate'] = {self.metrics_at_k[i]:HitRate[i].item() for i in range(len(self.metrics_at_k))};
        result['NDCG'] = {self.metrics_at_k[i]:NDCG[i].item() for i in range(len(self.metrics_at_k))};
        result['ReasonTime'] = t_reason/n_step;
        json_util.jsonsave(result, self.save_path + '/test_result.json');
        return self._report_result(result);


    def _report_batch_result(self, t_start, cur_step, n_step, hitrate, ndcg, qoe, trafficload):
        '''Report batch test result
        '''    
        str_show1 = '';
        for it in range(len(self.metrics_at_k)):
            str_show1 += f'@{self.metrics_at_k[it]:2}    {hitrate[it].item():.4f}           {ndcg[it].item():.4f}\n'
        str_show2 = '';
        for it in range(len(self.cache_size)):
            str_show2 += f'  {self.cache_size[it]:.1f}   -   {qoe[it]:.6f} -- {trafficload[it]:.6f}       \n'
        logger.info(
            f'batch test[{cur_step+1}/{n_step}] time consuming: {util.s2hms(time.time() - t_start)}\n'
            f'--------------------------------------\n'
            f'[{self.model.type}]Result of batch test\n'
            f'--------------------------------------\n'
            f'Overall performance of the model\n'
            f'--------------------------------------\n'
            f'         HR              NDCG         \n' + str_show1 +
            f'--------------------------------------\n'
            f'Performance report of the model\n'
            f'--------------------------------------\n'
            f'cache_size  -  QoE -- TrafficLoad     \n' + str_show2
        )

    def _report_result(self, result):
        '''Report the test result
        '''
        str_show1 = '';
        for it in self.metrics_at_k:
            str_show1 += f'@{it:2}       {result["HitRate"][it]:.4f}        {result["NDCG"][it]:.4f}\n'
        str_show2 = '';
        for it in self.cache_size:
            str_show2 += f'  {it:.1f}   -   {result["QoE"][it]:.6f} -- {result["TrafficLoad"][it]:.6f}       \n'
        str = f'[{self.model.type}]Result of test\n'\
             f'--------------------------------------\n'\
            f'Reasoning Time consuming:{result["ReasonTime"]}\n'\
            f'--------------------------------------\n'\
            f'Overall performance of the model\n'\
            f'--------------------------------------\n'\
            f'         HR              NDCG         \n' + str_show1 +\
            f'--------------------------------------\n'\
            f'Performance report of the model\n'\
            f'--------------------------------------\n'\
            f'cache_size  -  QoE -- TrafficLoad     \n' + str_show2;
        logger.info(str);
        return str;

class ConventionalTester(AbstractTester):
    '''Tester for conventional algorithm: FIFO,LRU,LFU

    Parameters:
    -----------
    
    '''
    def __init__(self, config, model) -> None:
        super().__init__(config['test'], model);
        self.cfg = config;
        if self.save:
            self.save_path, _ = os.path.split(self.model_save_path);
        else:
            self.save_path = './cache/unsaved_data/[' + util.get_date('_') + ']';
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path);
    
    def _caching_and_cal_qoe_trafficload(self, data, cache_size_list):
        '''Calculate QoE and Traffice Load at cache_size

        Parameters:
        -----------
        data:torch.Tensor
        (batch_size, req_len)

        cache_size_list:list

        Returns:
        --------
        QoE:dict

        TrafficLoad:dict
        '''
        batch_size, req_len = data.shape;
        #cache the item
        self.model.clear();
        for batch_id in range(batch_size):
            #su:(slide_len, T)
            su = data[batch_id, :].unfold(-1, self.slide_T + 1, self.slide_T + 1)[:, :self.slide_T];
            self.model.update(su.reshape(-1));
        QoE = {};
        TrafficLoad = {};
        for cache_size in cache_size_list:
            cache_num = int(np.round(self.bs_storagy * cache_size));
            #Get the cache corresponding to cache size
            cache_set = self.model.generate_subcache(cache_num);
            logger.debug('Tester._caching_and_cal_qoe_trafficload\n'
                            f'cache_num: {cache_num} - len(cache_set): {len(cache_set)}')
            #calculate qoe and trafficload
            qoe, userload, allload = 0, 0, 0;
            for batch_id in range(batch_size):
                #R:real data set
                R = set(data[batch_id, :].unfold(-1, self.slide_T + 1, self.slide_T + 1)[:, -1].tolist());
                if len(cache_set & R) > (req_len - data[batch_id, :].eq(0).sum().item())*self.cache_satisfaction_ratio:
                    qoe += 1;
                userload += len(R - cache_set);
                allload += len(R);
            QoE[cache_size] = qoe/batch_size;
            TrafficLoad[cache_size] = userload/allload;
        return QoE, TrafficLoad;



    
    def test(self, dataset):
        '''Test the conventional model:FIFO/LRU/LFU

        Notes:
        Conventional algorithms do not have predictive capabilities, 
        so hit rate and ndcg are no longer calculated here.
        '''
        #Adjust the number of workers of dataloader according to the system CPU and system
        if platform.system().lower() == 'linux':
            num_workers = self.cfg['linux_fast_num_workers'];
        else:
            num_workers = 0;
        result = {'QoE':{}, 'TrafficLoad':{}};
        #initial
        for cs in self.cache_size:
            result['QoE'][cs] = 0;
            result['TrafficLoad'][cs] = 0;
        test_loader = generator.get_dataloader(dataset, self.cfg['net'], num_workers, self.batch_size, shuffle = True, mode = 'test');
        n_step = int(np.ceil(dataset['u_num']/self.batch_size)) * 2;
        for i in range(n_step):
            t_reason = 0;
            t = time.time();
            qoe_batch, tl_batch = [], [];
            #test_data:(batch_size, seq_len)
            #next_req:(batch_size, 1)
            _, test_data, next_req = iter(test_loader).__next__();
            test_data, next_req = test_data.to(self.device), next_req.to(self.device);
            #new data:(batch_size, seq_len + 1)
            data = torch.cat((test_data, next_req.unsqueeze(-1)), dim = -1);
            #calculate qoe and traffic load
            qoe, trafficload = self._caching_and_cal_qoe_trafficload(data, self.cache_size);
            for cs in self.cache_size:
                result['QoE'][cs] += qoe[cs];
                qoe_batch.append(qoe[cs]);
                result['TrafficLoad'][cs] += trafficload[cs];
                tl_batch.append(trafficload[cs]);
            self._report_batch_result(t, i, n_step, qoe_batch, tl_batch);
        for cs in self.cache_size:
            result['QoE'][cs] /= n_step;
            result['TrafficLoad'][cs] /= n_step;
            result['ReasonTime'] = t_reason/n_step;
        json_util.jsonsave(result, self.save_path + '/test_result.json');
        return self._report_result(result);

    def _report_batch_result(self, t_start, cur_step, n_step, qoe, trafficload):
        '''Report batch test result
        '''    
        
        str_show2 = '';
        for it in range(len(self.cache_size)):
            str_show2 += f'  {self.cache_size[it]:.1f}   -   {qoe[it]:.6f} -- {trafficload[it]:.6f}       \n'
        logger.info(
            f'batch test[{cur_step+1}/{n_step}] time consuming: {util.s2hms(time.time() - t_start)}\n'
            f'--------------------------------------\n'
            f'[{self.model.type}]Result of batch test\n'
            f'--------------------------------------\n'
            f'--------------------------------------\n'
            f'Performance report of the model\n'
            f'--------------------------------------\n'
            f'cache_size  -  QoE -- TrafficLoad     \n' + str_show2
        )

    def _report_result(self, result):
        '''Report the test result
        '''
        str_show2 = ''
        for it in self.cache_size:
            str_show2 += f'  {it:.1f}   -   {result["QoE"][it]:.6f} -- {result["TrafficLoad"][it]:.6f}       \n'
        str = f'[{self.model.type}]Result of test\n'\
             f'--------------------------------------\n'\
            f'Reasoning Time consuming:{result["ReasonTime"]}\n'\
            f'--------------------------------------\n'\
            f'--------------------------------------\n'\
            f'Performance report of the model\n'\
            f'--------------------------------------\n'\
            f'cache_size  -  QoE -- TrafficLoad     \n' + str_show2;
        logger.info(str);
        return str;