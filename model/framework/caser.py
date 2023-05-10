# @Time   : 2023.03.03
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com

import torch
from lib import util, glb_var
from model.cnnnet import VerticalConv, HorizontalConv

class Caser(torch.nn.Module):
    '''Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding
        magic version
    Parameters:
    -----------
    net_cfg_dict:dict
    '''
    def __init__(self, net_cfg_dict, batch_size) -> None:
        super().__init__();
        util.set_attr(self, net_cfg_dict);
        self.encoder = torch.nn.Embedding(self.input_types, self.d);
        self.vrtconv = VerticalConv(batch_size, self.L, self.d, self.n_kernels);
        self.hrtconv = HorizontalConv(batch_size, self.L, self.d, self.n_kernels);
        self.dropout = torch.nn.Dropout(0.1);
        self.dense1 = torch.nn.Linear(2 * self.n_kernels, self.d);
        self.relu = torch.nn.ReLU();
        self.dense2 = torch.nn.Linear(2 * self.d, self.input_types);
        if net_cfg_dict['is_embed_net_manual_init']:
            self.apply(self._init_para);
    
    def _init_para(self, module):
        if isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=1/self.encoder.embedding_dim)
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_()
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, su):
        #su: (batch_size, slide_len, L)
        #Eu:(batch_size, d)
        Eu = self.encoder(su.reshape(su.shape[0], -1)).mean(dim = 1);
        #Ec:(batch_size, slide_len, L, d)
        Ec = self.encoder(su);
        #(batch_size, slide_len, n)
        vrt_o = self.vrtconv(Ec.transpose(0,1)).transpose(0,1).squeeze(-1);
        #(batch_size, slide_len, n)
        hrt_o = self.hrtconv(Ec.transpose(0,1)).transpose(0,1).squeeze(-2);
        #(batch_size, slide_len, d)
        conv_out = self.relu(self.dense1(self.dropout(torch.cat((vrt_o, hrt_o), dim = -1))));
        
        #(batch_size, slide_len, req_types)
        logits = self.dense2(torch.cat((conv_out, Eu.unsqueeze(1).repeat(1, conv_out.shape[1], 1)), dim = -1));
        return logits;

    def cal_loss(self, su, next_item):
        '''Calculate loss for Caser

        Parameters:
        -----------
        su:torch.Tensor
        (batch_size, slide_len, L)

        next_item:torch.Tensor
        (batch_size, slide_len)
        '''
        #pos_logits: (batch_size, slide_len, req_types)
        pos_logits = self.forward(su);
        loss = torch.nn.CrossEntropyLoss(ignore_index = self.mask_item[0])(pos_logits.reshape(-1, pos_logits.shape[-1]), next_item.reshape(-1));
        #Eu:(batch_size, d, 1)
        Eu = self.encoder(su.reshape(su.shape[0], -1)).mean(dim = 1).unsqueeze(-1);
        #(batch_size, neg_samples)
        neg_items = torch.randint(0, self.input_types, (Eu.size(0), self.neg_samples), device = glb_var.get_value('device'));
        #batch_size, neg_samples
        neg_logits = torch.matmul(self.encoder(neg_items), Eu)
        pos_loss = torch.nn.functional.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits));
        neg_loss = torch.nn.functional.binary_cross_entropy_with_logits(neg_logits, torch.ones_like(neg_logits));
        return loss + pos_loss + neg_loss;