import torch
from lib import util,glb_var
from model import attnet, loss
from data.generator import NextReqDataSet

class Encoder(torch.nn.Module):
    '''Encoder for EC4SRec

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

class EC4SRec(torch.nn.Module):
    '''Explanation Guided Contrastive Learning for Sequential Recommendation

    Parameters:
    -----------
    net_cfg_dict:dict
    configure of the net

    Refrence:
    ---------
    [1]WANG L, LIM E P, LIU Z, et al. 
    Explanation guided contrastive learning for sequential recommendation[C]//
    Proceedings of the 31st ACM International Conference on Information & Knowledge Management. 2022: 2017-2027.
    '''
    def __init__(self, net_cfg_dict) -> None:
        super().__init__();
        util.set_attr(self, net_cfg_dict);
        self.encoder = Encoder(self.d, self.d_fc, self.n_heads, self.n_layers, 
                               self.input_types, self.posenc_buffer_size, self.mask_item);
        self.loss_func = loss.BPRLoss(gamma = glb_var.get_value('eps'))
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

    def _impt_score_step(self, su):
        '''step for the loop of method[cal_impt_score]

        Parameters:
        -----------
        su:torch.Tensor
        (1, req_len)

        Returns:
        --------
        (req_len)
        '''
        su = su.to(glb_var.get_value('device'));
        #next_item_logits:(1, req_types)
        next_item_logits = self.forward(su);
        #s:(req_types, d)
        s = torch.autograd.grad(
            outputs = next_item_logits,
            inputs = self.encoder.seq_embed.weight,
            grad_outputs = torch.ones_like(next_item_logits)
        )[0];
        s[torch.isnan(s) | torch.isinf(s)] = glb_var.get_value('eps');
        #(req_len)
        return s[su, :].sum(dim = -1);


    def cal_impt_score(self, dataset):
        '''Calculate the important score for Explanation method

        Parameters:
        ----------
        dataset:dict

        Returns:
        --------
        impt_score_train:torch.Tensor
        (batch_size, seq_len_train)
        impt_score_valid:torch.Tensor
        (batch_size, seq_len_valid)

        '''
        #training important score
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache();
        #su_batch:(batch_size, req_len)
        _, su_batch, _ = iter(
            torch.utils.data.DataLoader(NextReqDataSet(dataset, mode = 'train'), batch_size = dataset['u_num'])
            ).__next__();
        impt_score_train = torch.zeros((dataset['u_num'], su_batch.shape[1])).to(glb_var.get_value('device'));
        for batch_idx in range(dataset['u_num']):
            impt_score_train[batch_idx, :] = self._impt_score_step(su_batch[[batch_idx], :].clone());
        
        #validation important score
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache();
        #su_batch:(batch_size, req_len)
        _, su_batch, _ = iter(
            torch.utils.data.DataLoader(NextReqDataSet(dataset, mode = 'valid'), batch_size = dataset['u_num'])
            ).__next__();
        impt_score_valid = torch.zeros((dataset['u_num'], su_batch.shape[1])).to(glb_var.get_value('device'));
        for batch_idx in range(dataset['u_num']):
            impt_score_valid[batch_idx, :] = self._impt_score_step(su_batch[[batch_idx], :].clone());
        glb_var.get_value('logger').info('Important score update complete.')
        return impt_score_train.clone(), impt_score_valid.clone();

    def cal_loss(self, su_batch, next_req_bacth, su_batch_operat_tuple, batch_idx):
        ''' Calculate the loss for training the model

        Parameters:
        -----------
        su_batch:torch.Tensor
        (batch_size, seq_len)

        next_req_batch:torch.Tensor
        (batch_size)

        su_batch_operat_tuple:tuple
        (3), item :list

        batch_idx:int

        Returns:
        --------
        loss
        '''
        #unpack
        su_opr_pos_list, su_opr_neg_list, su_opr_rtrl_list = su_batch_operat_tuple;
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
        #Self-Supervised Contrastive learning loss
        #SSL pos loss
        #hu_a:(1, d)
        hu_a1 = self.encoder(su_opr_pos_list[0][[batch_idx], :]);
        hu_a2 = self.encoder(su_opr_pos_list[1][[batch_idx], :]);
        #hu_opr_pos:(2*batch -2, d)
        hu_opr_pos = torch.cat(
            (
                self.encoder(su_opr_pos_list[0][torch.arange(batch_size) != batch_idx, :]),
                self.encoder(su_opr_pos_list[1][torch.arange(batch_size) != batch_idx, :])
            ),
            dim = 0
        );
        ssl_pos_score_pos = torch.matmul(hu_a1, hu_a2.transpose(0, 1));
        ssl_pos_score_neg = torch.matmul(hu_a1, hu_opr_pos.mean(dim = 0).unsqueeze(-1));
        ssl_pos_loss = self.loss_func(ssl_pos_score_pos, ssl_pos_score_neg);
        #SSL neg loss
        #hu_aneg:(1, d)
        hu_aneg = self.encoder(su_opr_neg_list[0][[batch_idx], :]);
        #hu_a_:(batch - 1, d)
        hu_a_ = self.encoder(su_opr_neg_list[0][torch.arange(batch_size) != batch_idx, :]);
        #hu_apos:(2*batch, d)
        hu_apos = torch.cat((self.encoder(su_opr_pos_list[0]), self.encoder(su_opr_pos_list[1])),dim = 0);
        
        ssl_neg_score_pos = torch.matmul(hu_aneg, hu_a_.mean(dim = 0).unsqueeze(-1));
        ssl_neg_score_neg = torch.matmul(hu_aneg, hu_apos.mean(dim = 0).unsqueeze(-1));
        ssl_neg_loss = self.loss_func(ssl_neg_score_pos, ssl_neg_score_neg);
        #Supervised Contrastive learning loss
        if su_opr_rtrl_list[batch_idx] is torch.nan:
            return loss_rec + self.lambda_sl_pos * ssl_pos_loss + self.lambda_sl_neg * ssl_neg_loss; 
        su_neg = su_batch[torch.arange(batch_size).to(glb_var.get_value('device'))!= batch_idx, :];
        for i in range(batch_size):
            if not torch.any(torch.as_tensor(su_opr_rtrl_list[i]).isnan()) and i != batch_idx:
                su_neg = torch.cat((su_neg, su_opr_rtrl_list[i]), dim = 0);
        #hu_rtrl:(1, d)
        hu_rtrl = self.encoder(su_opr_rtrl_list[batch_idx]);
        #su_neg:(new_batch_size, d)
        hu_neg = self.encoder(su_neg);
        
        sl_pos_score = torch.matmul(hu, hu_rtrl.transpose(0, 1))/self.tau;
        sl1_neg_score = torch.matmul(hu, hu_neg.mean(dim = 0).unsqueeze(-1))/self.tau;
        sl2_neg_score = torch.matmul(hu_rtrl, hu_neg.mean(dim = 0).unsqueeze(-1))/self.tau;

        loss_cl = self.loss_func(sl_pos_score, sl1_neg_score) + self.loss_func(sl_pos_score, sl2_neg_score);
        return loss_rec + self.lambda_cl * loss_cl + self.lambda_sl_pos * ssl_pos_loss + self.lambda_sl_neg * ssl_neg_loss;
