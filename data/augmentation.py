import torch
from lib import util, glb_var, callback
import numpy as np

class BasicAugmentation():
    '''Basic Augmentation for CL4SRec, Duo4SRec

    Example:
    --------
    >>> from data.augmentation import BasicAugmentation
    >>> ...
    >>> ba = BasicAugmentation(aug_cfg);
    >>> ba.sample_opr();
    >>> ba.update_scale(new_scale)
    >>> su_seq = ba.operte(su);
    '''
    def __init__(self, aug_cfg_dict) -> None:
        util.set_attr(self, aug_cfg_dict);
        #type:"Basic"
        #operator:["crop", "mask", "reorder", "retrieval"]
        #scale:[eta, gamma, beta, next_req_batch_tensor]
        #   next_req_batch_tensor:torch.Tensor
        #   (batch_size, req_len)
        #opr_sample_num
        #mask_to
        self.opr_dict = dict(
            crop = self.opr_crop,
            mask = self.opr_mask,
            reorder = self.opr_reorder,
            retrieval = self.opr_retireval
        );
        self.opr_num = len(self.operator);
        self.opr_sampled_idx = [];
    
    def update_scale(self, new_scale):
        '''Update scale of operter
        especially for operater retrieval to update next-req-list

        Parameters:
        -----------
        new_scale:list
        '''
        self.scale = new_scale;

    def sample_opr(self):
        '''Sample operator
        '''
        self.opr_sampled_idx = [];
        opr_idx = np.random.choice(self.opr_num, self.opr_sample_num, replace = False);
        for i in range(self.opr_sample_num):
            self.opr_sampled_idx.append(opr_idx[i])
    
    def sample_idx(self, length, sample_num, batch_size = 1, cont = False):
        '''Sample from length

        Parameters:
        -----------
        length:int

        sample_num:int

        batch_size:int, optional
        default:1

        cont:bool, optional
        True:The initial sampling of each batch is random, and the subsequent numbers are continuous
        False:All random samples of each batch
        default:False

        Reutrns:
        --------
        idx:torch.Tensor
        (batch_size, sample_num)

        batch_idx:torch.Tensor
        (batch_size, 1)
        '''
        if cont is False:
            idx = torch.zeros((0, sample_num)).to(torch.int64);
            for _ in range(batch_size):
                idx = torch.cat(
                    (
                    idx,
                    torch.from_numpy(np.random.choice(length, sample_num, replace = False)).to(torch.int64).unsqueeze(0)
                    ),
                    dim = 0
                )
            
        else:
            #idx:(batch_size, 1)
            idx = torch.from_numpy(np.random.choice(length - sample_num, batch_size, replace = True)).to(torch.int64).reshape(-1, 1);
            idx_copy = idx.clone();
            #tlb:(batch_size, length)
            tlb = torch.arange(length).unsqueeze(0).repeat(batch_size, 1);
            for i in range(1, sample_num):
                idx = torch.cat(
                    (
                        idx,
                        torch.gather(tlb, dim = -1, index = idx_copy + i)
                    ),
                    dim = -1
                )
        batch_idx = torch.arange(0, batch_size).reshape(-1, 1);
        return idx, batch_idx;

    def opr_crop(self, eta, su):
        '''Operator Random crop

        Parameters:
        -----------
        eta:float
        Scale of random crop

        su:torch.Tensor
        (batch_size, req_len)

        Returns:
        --------
        su_crop:torch.Tensor
        (batch_size, floor(req_len * eta))
        '''
        crop_num = int(np.floor(eta * su.shape[1]));
        crop_idx, batch_idx = self.sample_idx(su.shape[1], crop_num, batch_size = su.shape[0], cont = True);
        return su[batch_idx, crop_idx];

    def opr_mask(self, gamma, su):
        '''Operator Random mask

        Parameters:
        -----------
        gamma:float
        Scale of random mask

        su:torch.Tensor
        (batch_size, req_len)

        Returns:
        --------
        su_mask:torch.Tensor
        (batch_size, batch_size, req_len)
        '''
        mask_num = int(np.floor(gamma * su.shape[1]));
        mask_idx, batch_idx = self.sample_idx(su.shape[1], mask_num, batch_size = su.shape[0]);
        su[batch_idx, mask_idx] = self.mask_to;
        return su;

    def opr_reorder(self, beta, su):
        '''Operator reorder

        Parameters:
        -----------
        beta:float
        Scale of random mask

        su:torch.Tensor
        (batch_size, req_len)

        Returns:
        --------
        su_reorder:torch.Tensor
        (batch_size, batch_size, req_len)
        '''
        reorder_num = int(np.floor(beta * su.shape[1]));
        reorder_idx, batch_idx = self.sample_idx(su.shape[1], reorder_num, batch_size = su.shape[0]);
        sub_su = su[batch_idx, reorder_idx];
        shuffle_idx, _ = self.sample_idx(reorder_num, reorder_num, batch_size = su.shape[0]);
        sub_su = sub_su[batch_idx, shuffle_idx];
        su[batch_idx, reorder_idx] = sub_su;
        return su; 

    def opr_retireval(self, batch_idx, next_req_batch_tensor, su):
        '''Operator retrieval

        Parameters:
        -----------
        batch_idx:int

        next_req_batch_tensor:torch.Tensor
        (batch_size)

        su:torch.Tensor
        (batch_size, seq_len)

        '''
        #next_req_batch_tensor:(batch_size)
        idx_array = torch.arange(next_req_batch_tensor.shape[0]).to(glb_var.get_value('device'));
        bool_idx = next_req_batch_tensor.eq(next_req_batch_tensor[batch_idx]);
        if bool_idx.sum() == 1:
            #no sequence has the same next-request with seq(batch_idx)
            return torch.nan;
        else:
            bool_idx[batch_idx] = False;
            idx_array = idx_array[bool_idx];
            return su[idx_array[np.random.choice(idx_array.shape[0], 1)], :];
        
    def operate(self, su):
        '''Operate on user sequence using sampled operator

        Parameters:
        -----------

        su:torch.Tensor
        (batch_size, seq_len)

        Returns:
        ---------
        seq_su:list
        [su_ai, su_aj, ...]
        '''
        seq_su = [];
        for i in range(self.opr_sample_num):
            opr = self.operator[self.opr_sampled_idx[i]];
            if opr != 'retrieval':
                seq_su.append(self.opr_dict[opr](self.scale[self.opr_sampled_idx[i]], su.clone()));
            else:
                su_rtrl = [];
                for batch_idx in range(su.shape[0]):
                    su_rtrl.append(self.opr_retireval(batch_idx, self.scale[self.opr_sampled_idx[i]], su.clone()));
                seq_su.append(su_rtrl);
        return seq_su;

class EGA(BasicAugmentation):
    '''Explanation Guided Augmentation for EC4SRec
    '''
    def __init__(self, aug_cfg_dict) -> None:
        super().__init__(aug_cfg_dict)
        #type:"EGA"
        #operator:["crop", "mask", "reorder", "retrieval"]
        #scale:[eta, gamma, beta, next_req_batch_tensor]
        #   next_req_batch_tensor:torch.Tensor
        #   (batch_size, req_len)
        #opr_sample_num
        #mask_to
        if 'retrieval' in self.operator and self.operator[-1] != 'retrieval':
            glb_var.get_value('logger').error('Operator [retrieval] must be at the end of the list.\nHint:'
                'Donnot forget to change the [scale] when adjusting, otherwise unexpected results or unexpected errors will occur');
            callback.CustomException('AugmentationConfigError');

    def sample_opr(self):
        '''Sample operator
        '''
        self.opr_sampled_idx = [];
        opr_idx = np.random.choice(self.opr_num - 1, self.opr_sample_num - 1, replace = False);
        for i in range(self.opr_sample_num - 1):
            self.opr_sampled_idx.append(opr_idx[i])
        #"retrieval" must be included
        self.opr_sampled_idx.append(self.opr_num - 1)

    def opr_crop(self, eta, impt_score, su):
        '''Operator Random crop

        Parameters:
        -----------
        eta:float
        Scale of random crop

        impt_score:torch.Tensor
        (batch_size, req_len)

        su:torch.Tensor
        (batch_size, req_len)

        Returns:
        --------
        su_crop:torch.Tensor
        (batch_size, floor(req_len * eta))
        '''
        batch_size, req_len = su.shape;
        crop_num = int(np.floor(eta * req_len));
        batch_idx = torch.arange(0, batch_size).reshape(-1, 1);
        all_idx = torch.arange(0, req_len).unsqueeze(0).repeat(batch_size, 1).to(glb_var.get_value('device'));
        #crop_idx:(batch_size, crop_num)
        _, crop_idx = impt_score.topk(k = crop_num, largest = False, dim = -1, sorted = True);
        su_crop_neg = su[batch_idx, crop_idx];
        su_crop_pos = su[torch.bitwise_not((all_idx.reshape(batch_size, req_len, 1) == 
                          crop_idx.reshape(batch_size, 1, crop_num)).sum(dim = -1).to(torch.bool))].reshape(batch_size, -1);
        return su_crop_pos, su_crop_neg;

    def opr_mask(self, gamma, impt_score, su):
        '''Operator Random mask

        Parameters:
        -----------
        gamma:float
        Scale of random mask

        impt_score:torch.Tensor
        (batch_size, req_len)

        su:torch.Tensor
        (batch_size, req_len)

        Returns:
        --------
        su_mask:torch.Tensor
        (batch_size, batch_size, req_len)
        '''
        batch_size = su.shape[0];
        mask_num = int(np.floor(gamma * su.shape[1]));
        batch_idx = torch.arange(0, batch_size).reshape(-1, 1);
        su_mask_pos, su_mask_neg = su.clone(), su.clone();
        #mask_idx:(batch_size, mask_num)
        _, mask_pos_idx = impt_score.topk(k = mask_num, largest = False, dim = -1, sorted = True);
        _, mask_neg_idx = impt_score.topk(k = mask_num, largest = True, dim = -1, sorted = True);
        su_mask_pos[batch_idx, mask_pos_idx] = self.mask_to;
        su_mask_neg[batch_idx, mask_neg_idx] = self.mask_to;
        return su_mask_pos, su_mask_neg;

    def opr_reorder(self, beta, impt_score, su):
        '''Operator reorder

        Parameters:
        -----------
        beta:float
        Scale of random mask

        impt_score:torch.Tensor
        (batch_size, req_len)

        su:torch.Tensor
        (batch_size, req_len)

        Returns:
        --------
        su_reorder:torch.Tensor
        (batch_size, req_len)
        '''
        batch_size = su.shape[0];
        reorder_num = int(np.floor(beta * su.shape[1]));
        batch_idx = torch.arange(0, batch_size).reshape(-1, 1);
        #reorder_idx:(batch_size, reorder_num)
        _, reorder_idx = impt_score.topk(k = reorder_num, largest = False, dim = -1, sorted = True);
        sub_su = su[batch_idx, reorder_idx];
        shuffle_idx, _ = self.sample_idx(reorder_num, reorder_num, batch_size = batch_size);
        sub_su = sub_su[batch_idx, shuffle_idx];
        su[batch_idx, reorder_idx] = sub_su;
        #su_pos
        return su;

    def opr_retireval(self, batch_idx, next_req_batch_tensor, impt_score, su, eps = 1e-9):
        '''Operator retrieval

        Parameters:
        -----------
        batch_idx:int

        next_req_batch_tensor:torch.Tensor
        (batch_size)

        impt_score:torch.Tensor
        (batch_size, req_len)

        su:torch.Tensor
        (batch_size, seq_len)

        Returns:
        --------
        (1, req_len)
        '''
        #next_req_batch_tensor:(batch_size)
        idx_array = torch.arange(next_req_batch_tensor.shape[0]).to(glb_var.get_value('device'));
        bool_idx = next_req_batch_tensor.eq(next_req_batch_tensor[batch_idx]);
        if bool_idx.sum() == 1:
            #no sequence has the same next-request with seq(batch_idx)
            return torch.nan;
        elif bool_idx.sum() == 2:
            bool_idx[batch_idx] = False;
            return su[idx_array[[bool_idx]], :]
        else:
            bool_idx[batch_idx] = False;
            idx_array = idx_array[bool_idx];
            p = torch.zeros_like(idx_array);
            #su_tgt:(req_len)
            su_tgt = su[batch_idx, :];
            for i in range(p.shape[0]):
                su_k = su[idx_array[i], :];
                p[i] = impt_score[batch_idx, torch.isin(su_k, torch.as_tensor(list(set(su_tgt.tolist()) & set(su_k.tolist()))).to(glb_var.get_value('device')))].sum() *\
                    (len(set(su_tgt.tolist()) & set(su_k.tolist())) / len(set(su_tgt.tolist()) | set(su_k.tolist())));
            p = p + eps;
            p = p/p.sum();
            dist = torch.distributions.Categorical(p);
            return su[[dist.sample()], :];

    def operate(self, impt_score, su):
        '''Operate on user sequence using sampled EGA operator

        Parameters:
        -----------
        impt_score:torch.Tensor
        (batch_size, req_len)

        su:torch.Tensor
        (batch_size, req_len)

        Returns:
        --------
        seq_su_pos:list

        seq_su_neg:list
        hint:maybe empty when sample less than 3

        seq_su_rtrl:list
        '''
        seq_su_pos = [];
        seq_su_neg = [];
        seq_su_rtrl = [];
        for i in range(self.opr_sample_num):
            opr = self.operator[self.opr_sampled_idx[i]];
            if opr != 'retrieval' and opr != 'reorder':
                su_a_pos, su_a_neg = self.opr_dict[opr](self.scale[self.opr_sampled_idx[i]], impt_score.clone(), su.clone());
                seq_su_pos.append(su_a_pos);
                seq_su_neg.append(su_a_neg);
            elif opr == 'reorder':
                seq_su_pos.append(self.opr_dict[opr](self.scale[self.opr_sampled_idx[i]], impt_score.clone(), su.clone()));
            else:#'retrieval
                for batch_idx in range(su.shape[0]):
                    seq_su_rtrl.append(self.opr_retireval(batch_idx, self.scale[self.opr_sampled_idx[i]], impt_score, su.clone()));

        return seq_su_pos, seq_su_neg, seq_su_rtrl;
        

    

def get_augmentation(aug_cfg_dict):
    if aug_cfg_dict['type'].lower() == 'basic':
        return BasicAugmentation(aug_cfg_dict);
    elif aug_cfg_dict['type'].lower() == 'ega':
        return EGA(aug_cfg_dict);
    else:
        glb_var.get_value('logger').error(f'augmentation type [{aug_cfg_dict["type"]}] is not supported.');
        raise callback.CustomException('AugmentationTypeError');