# @Time   : 2023.03.03
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com

import torch
from lib import util,glb_var
from model import attnet, loss


class Encoder(torch.nn.Module):
    '''Encoder for Duo4SRec

    Parameters:
    -----------
    d:int
    Dimension of embedded victor

    d_q:int
    Dimension of Q matrix

    d_k:int
    Dimension of K matrix

    d_v:int
    Dimension of V matrix

    d_fc:int
    Number of nodes in the dense network
    
    n_heads:int
    Number of the head

    input_types:int

    posenc_buffer_size:int
    The length of the position embedding register

    mask_item:list

    is_norm_first:bool,optinal
    The sequence of attention network NORM
    '''
    def __init__(self, d, d_fc, n_heads, n_layers, 
                 input_types, posenc_buffer_size, mask_item, is_norm_first = False) -> None:
        super().__init__();
        if is_norm_first:
            encoderlayer = attnet.EncoderLayer_PreLN;
        else:
            encoderlayer = attnet.EncoderLayer_PostLN;
        self.mask_item = mask_item;
        self.seq_embed = torch.nn.Embedding(input_types, d, padding_idx = self.mask_item[0]);
        self.pos_embed = attnet.LearnablePositionEncoding(d, posenc_buffer_size);
        self.embed_dropout = torch.nn.Dropout(0.1);
        #bad performance: self.pos_embed = attnet.PositionEncoding(d, max_len = posenc_buffer_size)
        self.Layers = torch.nn.ModuleList(
            [encoderlayer(d, d_fc, n_heads) for _ in range(n_layers)]
        )

    def forward(self, enc_input):
        #enc_input:(batch_size, req_len)
        #enc_output:(batch_size, req_len, d)
        #mask:(batch_size, seq_len, seq_len)
        #enc_output = self.pos_embed(self.seq_embed(enc_input)) * self.seq_embed.embedding_dim ** 0.5;
        enc_output = self.seq_embed(enc_input)* self.seq_embed.embedding_dim ** 0.5;
        enc_output = enc_output + self.pos_embed(enc_input);
        enc_output = self.embed_dropout(enc_output);
        mask = torch.bitwise_or(
            attnet.attn_pad_msk(enc_input, enc_input.shape[1], mask_item = self.mask_item),
            attnet.attn_subsequence_mask(enc_input)
        )
        for layer in self.Layers:
            enc_output = layer(enc_output, mask);
        #return hu:(batch_size, d)
        return enc_output[:, -1, :];

class Duo4SRec(torch.nn.Module):
    '''Contrastive Learning for Representation Degeneration Problem in Sequential Recommendation

    Parameters:
    -----------
    net_cfg_dict:dict
    configure of the net

    Refrence:
    ---------
    [1]QIU R, HUANG Z, YIN H, et al. 
    Contrastive learning for representation degeneration problem in sequential recommendation[C]//
    Proceedings of the fifteenth ACM international conference on web search and data mining. 2022: 813-823.
    '''
    def __init__(self, net_cfg_dict) -> None:
        super().__init__();
        util.set_attr(self, net_cfg_dict);
        self.encoder = Encoder(self.d, self.d_fc, self.n_heads, self.n_layers, 
                               self.input_types, self.posenc_buffer_size, self.mask_item, self.is_norm_first);
        self.loss_func = loss.BPRLoss(gamma = glb_var.get_value('eps'));
        self.classfy_func = torch.nn.CrossEntropyLoss(ignore_index = self.mask_item[0]);
        if net_cfg_dict['is_embed_net_manual_init']:
            self.apply(self._init_para);
    
    def _init_para(self, module):
        if isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=1/self.encoder.seq_embed.embedding_dim)
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_()
            if module.bias is not None:
                module.bias.data.zero_()

    def predictor(self, hu):
        '''Predictor for Duo4SRec

        Parameters:
        -----------
        hu:torch.Tensor
        representation fo the user request
        (batch_size, d)

        Returns:
        --------
        next-req:predicted next request
        (batch_size)
        '''
        #hu:(batch_size, d)
        #M:(req_types, d)
        #R:(batch_size, req_types)
        R = torch.matmul(hu, self.encoder.seq_embed.weight.transpose(0, 1));
        #next-req:(batch_size, req_types)
        return R;

    def forward(self, su):
        #su:(batch_size, req_len)
        #hu:(batch_size, d)
        hu = self.encoder(su);
        #output:(batch_size, req_types)
        return self.predictor(hu);

    def cal_loss(self, su_batch, next_req_bacth, su_batch_operat_list, batch_idx):
        ''' Calculate the loss for training the model

        Parameters:
        -----------
        su_batch:torch.Tensor
        (batch_size, seq_len)

        next_req_batch:torch.Tensor
        (batch_size)

        su_batch_operat_list:list
        for item in su_batch_operat_list
        item:(batch_size, req_operate_size)

        batch_idx:int

        Returns:
        --------
        loss
        '''
        batch_size, _ = su_batch.shape;
        #user representation:(1, d)
        su = su_batch[batch_idx, :].unsqueeze(0);
        hu = self.encoder(su.clone());
        #target request representation:(1, d)
        next_req = next_req_bacth[batch_idx].unsqueeze(0);
        hv_tgt = self.encoder(next_req.unsqueeze(0));
        # generate item representation
        #h_reqs:(req_types - 1, d)
        h_neg_req = self.encoder(torch.arange(self.input_types).unsqueeze(-1).to(glb_var.get_value('device'))
                              [torch.arange(self.input_types).to(glb_var.get_value('device')) != next_req, :]); 

        #Recomendation loss
        rec_pos_score = torch.matmul(hu, hv_tgt.transpose(0, 1));
        rec_neg_score = torch.matmul(hu, h_neg_req.mean(dim = 0).unsqueeze(-1))
        loss_rec = self.loss_func(rec_pos_score, rec_neg_score);

        #Supervised Contrastive learning loss
        if su_batch_operat_list[0][batch_idx] is torch.nan:
            return loss_rec;
        su_neg = su_batch[torch.arange(batch_size).to(glb_var.get_value('device')) != batch_idx, :];
        for i in range(batch_size):
            if not torch.any(torch.as_tensor(su_batch_operat_list[0][i]).isnan()) and i != batch_idx:
                su_neg = torch.cat((su_neg, su_batch_operat_list[0][i]), dim = 0);
        #hu_rtrl:(1, d)
        hu_rtrl = self.encoder(su_batch_operat_list[0][batch_idx]);
        #su_neg:(new_batch_size, d)
        hu_neg = self.encoder(su_neg);
        
        sl_pos_score = torch.matmul(hu, hu_rtrl.transpose(0, 1))/self.tau;
        sl1_neg_score = torch.matmul(hu, hu_neg.mean(dim = 0).unsqueeze(-1))/self.tau;
        sl2_neg_score = torch.matmul(hu_rtrl, hu_neg.mean(dim = 0).unsqueeze(-1))/self.tau;

        loss_cl = self.loss_func(sl_pos_score, sl1_neg_score) + self.loss_func(sl_pos_score, sl2_neg_score);
        return loss_rec + self.lambda_sl * loss_cl;
