import torch
from lib import glb_var, util
from model.cnnnet import VerticalConv

class AttnNet(torch.nn.Module):
    ''' Attention Network for PSAC_gen

    Parameters:
    -----------

    d: int
        The length of the vector to which the embedding is converted
    

    Methods:
    --------
    forward(self, Ec):
        argin:
            -Ec:(batch_size, slide_len, L, d)
        argout:
            -attn:(batch_size, slide_len, 1, L *d)


    '''
    def __init__(self, d) -> None:
        super().__init__();
        #create weight
        self.Wi = torch.nn.Linear(d, d, bias = False);
        self.Wj = torch.nn.Linear(d, d, bias = False);
        self.V = torch.nn.Linear(d, 1, bias = False);
        #activation function
        self.tanh = torch.nn.Tanh();
        self.softmax = torch.nn.Softmax(dim = -2);

    def forward(self, Ec):
        _, slide_len, L, d = Ec.shape;
        #Ec: (batch_size, slide_len, L, d)
        #ec_wi: (batch_size, slide_len, L, d) -> (batch_size, slide_len, L, 1, d)
        ec_wi = self.Wi(Ec).unsqueeze(-2);
        #ec_wj: (batch_size, slide_len, L, d)
        ec_wj = self.Wj(Ec);

        #Add the corresponding rows in each slide_len
        #ec_wi: (batch_size, slide_len, L, 1, d) -transpose-> (batch_size, L, slide_len, 1, d) -transpose-> (batch_size, L, 1, slide_len, d)
        #   -reshape-> (L, 1, slide_len * d * batch_size)
        #ec_wj: (batch_size, slide_len, L, d) -transpose-> (batch_size, L, slide_len, d) -reshape-> (L, slide_len*d*batch_size)
        #result:(L, L, slide_len*d) -reshape-> (L, L, slide_len, d) -transpose-> (L, slide_len, L, d)
        #   -transpose-> (slide_len, L, L, d) -unsqueeze-> (1, slide_len, L, L, d)
        ec_add = (ec_wi.transpose(1, 2).transpose(2, 3).reshape(L, 1, -1) \
            + ec_wj.transpose(1, 2).reshape(L,  -1))\
            .reshape(-1, L, L, slide_len, d).transpose(2, 3).transpose(1, 2);
        #w: (batch_size, slide_len, L, L, d)
        w = self.tanh(ec_add);
        #S: (batch_size, slide_len, L, L, 1)
        S = self.V(w);
        #alpha: (batch_size, slide_len, L, L, 1)
        alpha = self.softmax(S);
        #attn_i:  (batch_size, slide_len, L, L, d)
        attn_i = alpha.repeat(1, 1, 1, 1, d) * Ec.unsqueeze(-2).repeat(1, 1, 1, L, 1);
        #attn:(batch_size, slide_len, L, d)
        attn = attn_i.sum(dim = -2)
        return attn.reshape(-1, slide_len, 1, L * d)
    
class LSTFcNet(torch.nn.Module):
    '''Long short-term memory linear fully connected network for PSAC_gen

    Parameters:
    -----------

    d: int
        The length of the vector to which the embedding is converted

    req_set_len: int
        The length of the request set
    
    num_kernels: int, optional
        the numbers of the kernels
        default: 16
    
    L: int, optional
        the part of the sliding_window length
        default: 3

    Methods:
    --------

    forward(self, conv_output, attn, Eu):
        argin:
            -conv_output: (1, slide_len, 1, n)
            -attn: (1, slide_len, 1, L*d)
            -Eu: (1, 1, d)
        argout:
            -output: (1, slide_len, T, req_set_len)

    '''
    def __init__(self, d, req_set_len, num_kernels = 16, L = 3) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(num_kernels + L * d, d, bias = True);
        self.relu = torch.nn.ReLU();
        self.softmax = torch.nn.Softmax(dim = -1);
        self.fc2 = torch.nn.Linear(2 * d, req_set_len);
        self.dropout = torch.nn.Dropout(0.1);
    
    def forward(self, conv_output, attn, Eu):
        #conv_output:(batch_size, slide_len, 1, n)
        #attn:(batch_size, slide_len, 1, L*d)
        #fc1_input:(batch_size, slide_len, 1, n + L*d)
        fc1_input = self.dropout(torch.cat((conv_output, attn), dim = -1));
        #fc1_output:(batch_size, slide_len, 1, d)
        fc1_output = self.relu(self.fc1(fc1_input));
        #Eu:(batch_size, 1, d)
        #S: (batch_size, slide_len, d, d)
        S = torch.matmul(Eu.unsqueeze(1).repeat(1, fc1_output.shape[1], 1, 1).transpose(-1, -2), fc1_output);
        #dim = -1, normalization
        #alpha: (batch_size, slide_len, d, d)
        alpha = self.softmax(S);

        _, slide_len, d, _ = alpha.shape;

        #calculate longterm content attention
        #u_attn:  (batch_size, slide_len, d, d) -sum-> (batch_size, slide_len, d)
        u_attn = (alpha * fc1_output.repeat(1, 1, d, 1)).sum(dim = -1);

        #fc2_intput: (batch_size, slide_len, 2 * d)
        fc2_intput = torch.cat((Eu.repeat(1, slide_len, 1), u_attn), dim = -1);
        #fc2_output: (batch_size, slide_len, req_set_len)
        fc2_output = self.fc2(fc2_intput);

        return fc2_output;

def subsequence_attn_mask(n):
    '''Return the subseqence attention mask

    Parameters:
    -----------

    n: int
        Size of square mask
    
    Returns:
    --------

    mask: torch.Tensor
        subseeqence mask
    '''
    return torch.triu(torch.ones((n, n)), diagonal = 1).eq(1).to(glb_var.get_value('device'));

class PSAC_gen(torch.nn.Module):
    ''' Proactive Sequence-Aware Content Caching Net work for general data
        magic version
    Parameters:
    -----------
    net_cfg_dict:dict
    

    '''
    def __init__(self, net_cfg_dict, batch_size) -> None:
        super().__init__();
        util.set_attr(self, net_cfg_dict);
        self.encoder = torch.nn.Embedding(self.input_types, self.d);
        #self.VrtConv = VrtConv(self.d, batch_size, self.n_kernels, self.L);
        self.VrtConv = VerticalConv(batch_size, self.L, self.d, self.n_kernels)
        self.self_attn = AttnNet(self.d);
        self.LSTFcNet = LSTFcNet(self.d, self.input_types, self.n_kernels, self.L);
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
        #Ec: (batch_size, slide_len, L, d)
        Ec = self.encoder(su);
        #Eu: (batch_size, 1, d)
        Eu = self.encoder(su.reshape(su.shape[0], -1)).mean(dim = 1).unsqueeze(1);
        #o: (batch_size, slide_len, 1, n)
        o = self.VrtConv(Ec.transpose(0,1)).transpose(0,1).transpose(-1, -2);
        #attn: (batch_size, slide_len, 1, L*d)
        attn = self.self_attn(Ec);
        #pro_logits: (batch_size, slide_len, req_set_len)
        pro_logits = self.LSTFcNet(o, attn, Eu);
        return pro_logits
    
    def cal_loss(self, su, next_item):
        '''Calculate loss for PSAC_gen
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