import torch
from lib import util,glb_var
from model import attnet, loss
from data.generator import RecDataSet

class Encoder(torch.nn.Module):
    '''Encoder for CL4SRec

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
        self.seq_embed = torch.nn.Embedding(input_types, d);
        self.pos_enc = attnet.PositionEncoding(d, max_len = posenc_buffer_size)
        self.Layers = torch.nn.ModuleList(
            [encoderlayer(d, d_fc, n_heads) for _ in range(n_layers)]
        )

    
    def forward(self, enc_input):
        #enc_input:(batch_size, req_len)
        #enc_output:(batch_size, req_len, d)
        #mask:(batch_size, seq_len, seq_len)
        enc_output = self.pos_enc(self.seq_embed(enc_input));
        mask = torch.bitwise_or(
            attnet.attn_pad_msk(enc_input, enc_input.shape[1], mask_item = self.mask_item),
            attnet.attn_subsequence_mask(enc_input)
        )
        for layer in self.Layers:
            enc_output = layer(enc_output.clone(), mask);
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
        self.loss_func = loss.BPRLoss()
        self.classfy_func = torch.nn.CrossEntropyLoss(ignore_index = self.mask_item[0]);

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

    def _impt_score_step(self, su, batch_idx, impt_score):
        '''step for the loop of method[cal_impt_score]
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
        impt_score[batch_idx, :] = s[su, :].sum(dim = -1);
        impt_score[batch_idx, :] = impt_score[batch_idx, :] / impt_score[batch_idx, :].sum();
        return impt_score;


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
            torch.utils.data.DataLoader(RecDataSet(dataset, mode = 'train'), batch_size = dataset['u_num'], shuffle = True)
            ).__next__();
        impt_score_train = torch.zeros((dataset['u_num'], su_batch.shape[1])).to(glb_var.get_value('device'));
        for batch_idx in range(dataset['u_num']):
            impt_score_train = self._impt_score_step(su_batch[[batch_idx], :].clone(), batch_idx, impt_score_train.clone());
        
        #validation important score
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache();
        #su_batch:(batch_size, req_len)
        _, su_batch, _ = iter(
            torch.utils.data.DataLoader(RecDataSet(dataset, mode = 'valid'), batch_size = dataset['u_num'], shuffle = True)
            ).__next__();
        impt_score_valid = torch.zeros((dataset['u_num'], su_batch.shape[1])).to(glb_var.get_value('device'));
        for batch_idx in range(dataset['u_num']):
            impt_score_valid = self._impt_score_step(su_batch[[batch_idx], :].clone(), batch_idx, impt_score_valid.clone());
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
        batch_size, seq_len = su_batch.shape;
        #user representation:(1, d)
        su = su_batch[batch_idx, :].unsqueeze(0);
        hu = self.encoder(su.clone());
        #target request representation:(1, d)
        next_req = next_req_bacth[batch_idx].unsqueeze(0);
        hv_tgt = self.encoder(next_req.unsqueeze(0));
        # generate item representation
        #h_reqs:(req_types, d)
        h_reqs = self.encoder(torch.arange(self.input_types).unsqueeze(-1).to(glb_var.get_value('device'))); 
        #h_neg_req:(req_types - 1, d)
        h_neg_req = h_reqs[torch.arange(h_reqs.shape[0]).to(glb_var.get_value('device')) != next_req, :];

        #Classified loss
        next_req_pre = self.forward(su);
        loss_cls = self.classfy_func(next_req_pre, next_req);

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
        hu_aneg = self.encoder(su_opr_neg_list[0][[batch_idx], :]);
        hu_a_ = self.encoder(su_opr_neg_list[0][torch.arange(batch_size) != batch_idx, :]);
        h1 = torch.cat((self.encoder(su_opr_pos_list[0]),self.encoder(su_opr_pos_list[1])),dim = 0);
        alpha1 = h1.shape[0] + batch_size - 1;
        h2 = self.encoder(su_opr_neg_list[0][torch.arange(batch_size) != batch_idx, :]);
        alpha2 = h2.shape[0];
        h = torch.cat((h1 * alpha1, h2 * alpha2), dim = 0).sum(dim = 0) / (alpha1 + alpha2);
        
        ssl_neg_loss =  - ((torch.matmul(hu_aneg, hu_a_.mean(dim = 0).unsqueeze(-1)) 
                                     - torch.matmul(hu_aneg, h.unsqueeze(-1))))/(batch_size - 1);
        #Supervised Contrastive learning loss
        su_neg = torch.zeros((0, seq_len)).to(glb_var.get_value('device'));
        for i in range(batch_size):
            if not torch.any(torch.as_tensor(su_opr_rtrl_list[i]).isnan()):
                su_neg = torch.cat((su_neg, su_opr_rtrl_list[i]), dim = 0);
        if su_opr_rtrl_list[batch_idx] is torch.nan or su_neg.shape[0] == 0:
            return loss_rec + self.lambda_cls * loss_cls + self.lambda_sl_pos * ssl_pos_loss + self.lambda_sl_neg * ssl_neg_loss; 
        #hu_rtrl:(1, d)
        hu_rtrl = self.encoder(su_opr_rtrl_list[batch_idx]);
        #su_neg:(new_batch_size, d)
        hu_neg = self.encoder(su_neg.to(torch.int64));
        
        sl_pos_score = torch.matmul(hu, hu_rtrl.transpose(0, 1))/self.tau;
        sl1_neg_score = torch.matmul(hu, hu_neg.mean(dim = 0).unsqueeze(-1))/self.tau;
        sl2_neg_score = torch.matmul(hu_rtrl, hu_neg.mean(dim = 0).unsqueeze(-1))/self.tau;

        loss_cl = self.loss_func(sl_pos_score, sl1_neg_score) + self.loss_func(sl_pos_score, sl2_neg_score);
        return loss_rec + self.lambda_cl * loss_cl + self.lambda_cls * loss_cls + self.lambda_sl_pos * ssl_pos_loss + self.lambda_sl_neg * ssl_neg_loss;
