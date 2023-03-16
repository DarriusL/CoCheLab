import torch, os, time
from lib import util, glb_var, callback, json_util
import numpy as np
from data import augmentation, generator
import matplotlib.pyplot as plt
from collections import Counter

def get_save_path(cfg):
    if cfg['net']['is_norm_fist']:
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
        self.device = glb_var.get_value('device');
        self.logger = glb_var.get_value('logger');
        self.model = model.to(self.device);
        if self.save:
            self.save_path, _ = os.path.split(self.model_save_path);
        else:
            self.save_path = './cache/unsaved_data/[' + util.get_date('_') + ']';
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
    
    def train(self, train_data):
        self.logger.error('Method needs to be called after being implemented');
        raise callback.CustomException('NotImplementedError');

    def _check_nan(self, loss):
        if torch.isnan(loss):
            self.logger.error('loss is nan');
            raise callback.CustomException('ValueError');

    def _save(self, epoch):
        '''Save the model and configure
        '''
        torch.save(
            {"config":self.cfg, "model":self.model, "epoch":epoch},
            self.save_path + '/model.model'
        )
        self.logger.info(f'Save path: {self.save_path}')

class AbstractTester():
    '''Abstract parent trainer class
    '''
    def __init__(self, test_cfg_dict, model) -> None:
        util.set_attr(self, test_cfg_dict);
        self.device = glb_var.get_value('device');
        self.logger = glb_var.get_value('logger');
        self.model = model.to(self.device);

    def test(self, test_data):
        self.logger.error('Method needs to be called after being implemented');
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
        self.train_loss = [];
        self.valid_loss = [];
        self.valid_min_loss = np.inf if self.metric_less else -np.inf;
        self.cfg = config;

        if self.model.is_cl_method:
            #augmentation
            self.aug = augmentation.get_augmentation(config['augmentation']);

        #optimizer
        if self.optimizer_type.lower() == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate, betas = self.betas, weight_decay = self.weight_decay);
        elif self.optimizer_type.lower() == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = self.learning_rate, betas = self.betas, weight_decay = self.weight_decay);
        else:
            self.logger.warning(f"Unrecognized optimizer[{self.optimizer_type.lower()}], set default Adam optimizer");
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate);

        #lr decay
        if self.use_lr_schedule:
            self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer = self.optimizer,
                total_steps = self.max_epoch,
                max_lr = self.lr_max
            )

    def _run_epoch(self, data, mode = 'train', is_cl_method = True):
        ''' epcoh for trian and validation
        Parameters:
        ----------

        data:tuple
        (index, su_batch, next_req_bacth)
        >su_batch:(batch_size, seq_len)
        >next_req_bacth:(batch_size)

        mode:str,optional
        'train': backward
        'valid':no_grad

        Returns:
        --------
        loss
        '''

        index, su_batch, next_req_bacth = data;
        su_batch, next_req_bacth = su_batch.to(self.device), next_req_bacth.to(self.device);
        epoch_loss = [];
        if is_cl_method:
            #augmentation
            if self.model.type.lower() == 'duo4srec':
                self.aug.update_scale([next_req_bacth])
            elif self.model.type.lower() == 'ec4srec':
                self.aug.scale[-1] = next_req_bacth;
            self.aug.sample_opr();

            if self.model.type.lower() == 'ec4srec':
                #su_batch_operat_list:list
                if mode == 'train':
                    su_batch_operat_list = self.aug.operate(self.impt_score_train[index, :].clone(), su_batch.clone());
                elif mode == 'valid':
                    su_batch_operat_list = self.aug.operate(self.impt_score_valid[index, :].clone(), su_batch.clone());
                else:
                    self.logger.error('Unsupported mode type');
                    raise callback.CustomException('ModeError');
            else:
                #su_batch_operat_list:list
                su_batch_operat_list = self.aug.operate(su_batch.clone());

            for batch_idx in range(self.batch_size):

                #calculate loss
                loss = self.model.cal_loss(su_batch, next_req_bacth, su_batch_operat_list, batch_idx);

                if mode == 'train':
                    self._check_nan(loss);
                    loss.backward();
                    self.optimizer.step();
                    self.optimizer.zero_grad()
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache();
                epoch_loss.append(loss.item());
        else:
            #calculate loss
            loss = self.model.cal_loss(su_batch, next_req_bacth);

            if mode == 'train':
                self._check_nan(loss);
                loss.backward();
                self.optimizer.step();
                self.optimizer.zero_grad()
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache();

            epoch_loss.append(loss.item());
        return np.mean(epoch_loss)

    def _train_epoch(self, train_data):
        '''Train the model in one epoch
        
        Parameters:
        ----------

        train_data:tuple
        (su, next_req)
        >su_batch:(batch_size, seq_len)
        >next_req_bacth:(batch_size)

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

        valid_data:tuple
        (su, next_req)
        >su_batch:(batch_size, seq_len)
        >next_req_bacth:(batch_size)

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
        #initial important score
        if self.cfg['net']['type'].lower() == 'ec4srec':
            self.impt_score_train = (torch.ones((dataset['u_num'], self.cfg['dataset']['limit_length'])) / \
                self.cfg['dataset']['limit_length']).to(self.device);
            self.impt_score_valid = (torch.ones((dataset['u_num'], self.cfg['dataset']['limit_length'] + 1)) / \
                self.cfg['dataset']['limit_length']).to(self.device);
        loader_batch = self.batch_size;
        train_loader = torch.utils.data.DataLoader(generator.RecDataSet(dataset, mode = 'train'), batch_size = loader_batch, shuffle = True);
        valid_loader = torch.utils.data.DataLoader(generator.RecDataSet(dataset, mode = 'valid'), batch_size = loader_batch, shuffle = True);
        for epoch in range(self.max_epoch):
            if self.cfg['net']['type'].lower() == 'ec4srec':
                #if it's EC4SRec and meet the update imptscore strp
                if (epoch + 1)%self.model.impt_score_step == 0:
                    self.impt_score_train, self.impt_score_valid = self.model.cal_impt_score(dataset);
            train_data = iter(train_loader).__next__();
            #train
            self.train_loss.append(self._train_epoch(train_data));
            self.logger.info(f'[{self.model.type}]-[train]\n'
                                f'[epoch: {epoch + 1}/{self.max_epoch}] - train loss:{self.train_loss[-1]:.8f} - '
                                f'lr:{self.optimizer.param_groups[0]["lr"]}');
            if self.use_lr_schedule:
                self.lr_scheduler.step();
            if (epoch + 1)%self.valid_step == 0:
                valid_data = iter(valid_loader).__next__();
                #valid
                self.valid_loss.append(self._valid_epoch(valid_data));
                if self.metric_less:
                    #less, better
                    if self.valid_loss[-1] < self.valid_min_loss:
                        self._save(epoch)
                        self.valid_min_loss = self.valid_loss[-1];
                else:
                    #bigger, better
                    if self.valid_loss[-1] > self.valid_min_loss:
                        self._save(epoch)
                        self.valid_min_loss = self.valid_loss[-1];
                self.logger.info(f'[{self.model.type}]-[valid]\n'
                                 f'[epoch: {epoch + 1}/{self.max_epoch}]- valid loss:{self.valid_loss[-1]:.8f} - '
                                 f'valid min loss: {self.valid_min_loss:.8f}');
            
        plt.figure(figsize = (10, 6));
        plt.plot(np.arange(0, len(self.train_loss)) + 1, self.train_loss, label = 'train loss');
        plt.plot(np.arange(self.valid_step - 1, len(self.train_loss), self.valid_step) + 1, self.valid_loss, label = 'valid loss');
        plt.xlabel('epoch');
        plt.ylabel('loss');
        plt.yscale('log');
        plt.legend(loc='upper right')
        plt.savefig(self.save_path + '/loss.png', dpi = 400);

class Tester(AbstractTester):
    '''Tester for general algorithm of Contrastive Learning

    Parameters:
    -----------
    config:dict

    model:torch.nn.Module
    '''
    def __init__(self, config, model) -> None:
        super().__init__(config['test'], model);
        self.model.eval();
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
        #Representation of prediction requests:(batch_size, d)
        h_batch_pre = self.model.encoder(next_req_batch_pre.unsqueeze(-1));
        hr = torch.zeros((len(at_k))).to(self.device);
        ndcg = torch.zeros_like(hr).to(self.device);
        for batch_idx in range(batch_size):
            #uninteract:(uninteract_num)
            uninteract = torch.as_tensor(list(set(torch.arange(self.model.input_types).tolist()) - \
                set(su_batch[batch_idx, :].tolist()))).to(self.device);
            #cache_reqs:(100, 1)
            cach_reqs = torch.cat(
                (next_req_batch[[batch_idx]], 
                 uninteract[np.random.choice(uninteract.shape[0], 99, replace = False)]),
                 dim = 0
            ).unsqueeze(-1);
            #Representation of cache_reqs:(100, d)
            h_cache_reqs = self.model.encoder(cach_reqs);
            #scores:(100):Ascending
            scores = torch.matmul(h_cache_reqs, h_batch_pre[[batch_idx], :].transpose(0, 1)).squeeze(-1);
            rank = torch.arange(scores.shape[0]);
            rank[torch.argsort(scores, stable=True, descending=True)] = rank.clone();
            #count hit number
            hr[rank[0] < torch.as_tensor(at_k)] += 1;
            #add ndcg
            ndcg[rank[0] < torch.as_tensor(at_k)] += 1/np.log2(rank[0] + 2);
        hr = hr/batch_size;
        ndcg = ndcg/batch_size;
        return hr, ndcg;
            
    def _slide_item_cache(self, su, alter_dict):
        '''

        Parameters:
        -----------
        su:torch.Tensor
        (1, req_len)

        alter_dict:dict
        '''
        #(alter_topk)
        _, logits_topk = self.model(su).topk(self.alter_topk, dim = -1);
        alter_topk = logits_topk.squeeze(0).tolist();
        for i in range(self.alter_topk):
            if alter_topk[i] in alter_dict:
                alter_dict[alter_topk[i]] += 1;
            else:
                alter_dict[alter_topk[i]] = 1;
    
    def _caching_and_cal_qoe_trafficload(self, data, cache_size):
        '''Calculate QoE and Traffice Load at cache_size

        Parameters:
        -----------
        data:torch.Tensor
        (batch_size, req_len)

        cache_size:float

        Returns:
        --------
        qoe:float

        trafficload:flaot
        '''
        batch_size, req_len = data.shape;
        cache_num = int(np.round(self.bs_storagy * cache_size));
        #Get alternative cache dict
        alter_dict = {};
        for batch_id in range(batch_size):
            #su:(slide_len, T)
            su = data[batch_id, :].unfold(-1, self.slide_T + 1, self.slide_T)[:, :self.slide_T + 1];
            for i in range(su.shape[0]):
                self._slide_item_cache(su[:i+1, :].reshape(1, -1), alter_dict);
        #cache
        if len(alter_dict) <= cache_num:
            cache_set = set(alter_dict.keys());
        else:
            #choose top-cache_num
            chosen_list = sorted(alter_dict.items(), key = lambda x:x[1], reverse=True);
            cache_set = set();
            for i in range(cache_num):
                cache_set.add(chosen_list[i][0]);
        self.logger.debug('Tester._caching_and_cal_qoe_trafficload\n'
                          f'cache_num: {cache_num} - len(cache_set) = {len(cache_set)}')
        #calculate qoe and trafficload
        qoe, userload, allload = 0, 0, 0;
        for batch_id in range(batch_size):
            #R:real data set
            R = set(data[batch_id, :].unfold(-1, self.slide_T + 1, self.slide_T)[:, -1].tolist());
            if len(cache_set & R) > (req_len - data[batch_id, :].eq(0).sum().item())*self.cache_satisfaction_ratio:
                qoe += 1;
            userload += len(R - cache_set);
            allload += len(R);
        return qoe/batch_size, userload/allload;

    @torch.no_grad()
    def test(self, dataset):
        '''Test the trained model

        Parameters:
        -----------
        dataset:dict
        Processed dataset
        
        dataset_raw:dict
        original dataset
        '''
        result = {'HitRate':{}, 'NDCG':{}, 'QoE':{}, 'TrafficLoad':{}};
        #initial
        for cs in self.cache_size:
            result['QoE'][cs] = 0;
            result['TrafficLoad'][cs] = 0;
        
        test_loader = torch.utils.data.DataLoader(
            generator.RecDataSet(dataset, mode = 'test'), 
            batch_size = self.batch_size, 
            shuffle = False);
        n_step = int(np.ceil(dataset['u_num']/self.batch_size));
        HitRate, NDCG= torch.zeros(len(self.metrics_at_k)).to(self.device), torch.zeros(len(self.metrics_at_k)).to(self.device);
        for i in range(n_step):
            t = time.time();
            qoe_batch, tl_batch = [], [];
            #test_data:(batch_size, seq_len)
            #next_req:(batch_size, 1)
            _, test_data, next_req = iter(test_loader).__next__();
            test_data, next_req = test_data.to(self.device), next_req.to(self.device);
            #next_req_logits:(batch_size, req_types)
            next_req_logits = self.model(test_data);
            #next_req:(batch_size)
            next_req_pre = next_req_logits.argmax(dim = -1);
            self.logger.debug(f'pre_types:{len(Counter(next_req_pre.tolist()))}')
            #calculate hit rate and ndcg
            hitrate, ndcg = self._cal_hitrate_and_ndcg_atk(test_data.clone(), next_req_pre.clone(), next_req.clone(), self.metrics_at_k);
            HitRate += hitrate;
            NDCG += ndcg;
            #new data:(batch_size, seq_len)
            data = torch.cat((test_data, next_req.unsqueeze(-1)), dim = -1);
            #calculate qoe and traffic load
            for cs in self.cache_size:
                qoe, trafficload = self._caching_and_cal_qoe_trafficload(data.clone(), cs);
                result['QoE'][cs] += qoe;
                qoe_batch.append(qoe);
                result['TrafficLoad'][cs] += trafficload;
                tl_batch.append(trafficload);
            self._report_batch_result(t, i, n_step, hitrate, ndcg, qoe_batch, tl_batch);
        HitRate /= n_step;
        NDCG /= n_step;
        for cs in self.cache_size:
            result['QoE'][cs] /= n_step;
            result['TrafficLoad'][cs] /= n_step;
        result['HitRate'] = {self.hitrate_at_k[i]:HitRate[i].item() for i in range(len(self.metrics_at_k))};
        result['NDCG'] = {self.ndcg_at_k[i]:NDCG[i].item() for i in range(len(self.metrics_at_k))};

        json_util.jsonsave(result, self.save_path + '/test_result.json');
        self._report_result(result);


    def _report_batch_result(self, t_start, cur_step, n_step, hitrate, ndcg, qoe, trafficload):
        '''Report batch test result
        '''    
        str_show1 = '';
        for it in range(len(self.metrics_at_k)):
            str_show1 += f'@{self.metrics_at_k[it]:2}    {hitrate[it].item():.4f}           {ndcg[it].item():.4f}\n'
        str_show2 = '';
        for it in range(len(self.cache_size)):
            str_show2 += f' {self.cache_size[it]:.1f}    -     {qoe[it]:.4f} -- {trafficload[it]:.4f}       \n'
        self.logger.info(
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
        for it in self.metrics:
            str_show1 += f'@{it:2}       {result["HitRate"][it]:.4f}        {result["NDCG"][it]:.4f}\n'
        str_show2 = '';
        for it in self.cache_size:
            str_show2 += f' {it:.1f}   -    {result["QoE"][it]:.4f} -- {result["TrafficLoad"][it]:.4f}       \n'
        self.logger.info(
            f'[{self.model.type}]Result of test\n'
            f'--------------------------------------\n'
            f'Overall performance of the model\n'
            f'--------------------------------------\n'
            f'         HR              NDCG         \n' + str_show1 +
            f'--------------------------------------\n'
            f'Performance report of the model\n'
            f'--------------------------------------\n'
            f'cache_size  -  QoE -- TrafficLoad     \n' + str_show2
        )