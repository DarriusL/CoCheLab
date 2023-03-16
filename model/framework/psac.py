import torch
from lib import glb_var, util

class VrtConv(torch.nn.Module):
    ''' Vertical Convolutional Network for PSAC_gen

    Parameters:
    -----------

    d: int
        The length of the vector to which the embedding is converted
    
    num_kernels: int, optional
        the numbers of the kernels
        default: 16
    
    L: int, optional
        the part of the sliding_window length
        default: 3
    
    Methods:
    --------

    forward(self, Ec):
        argin:
            -Ec: (batch_size, slide_len, L, d)
        argout:
            -output: (batch_size, slide_len, 1, num_kernels)

    '''
    def __init__(self, d, batch_size, num_kernels = 16, L = 3) -> None:
        super().__init__();
        self.L = L;
        self.num_kernels = num_kernels;
        #calculate h repeat times
        rpt = num_kernels//L;
        endptr = num_kernels%L;
        h = list(range(1, L + 1));
        #create conv layers
        #   take default parameter as example
        #   there're 3 kind of height of kernels(h): 1, 2, 3
        #   output: (L-h+1, 1), so 3 maxpool is need
        layers = [torch.nn.Conv2d(batch_size, batch_size, (h[j], d)) for _ in range(rpt) for j in range(L)];
        for i in range(endptr):
            layers.append(torch.nn.Conv2d(batch_size, batch_size, (h[i], d)));
        self.convlayers = torch.nn.ModuleList(layers);
        #create maxpool layers
        h = list(map(lambda x: L-x+1, h));
        self.maxpool_layers = torch.nn.ModuleList([torch.nn.MaxPool2d((h[j], 1), stride = 1) for j in range(L)]);
        #activation function
        self.relu = torch.nn.ReLU();
    
    def forward(self, Ec):
        #officeal format: (batch, channal, height, width)
        
        #Ec: (batch_size, slide_len, L, d)
        #transpose Ec to (slide_len, batch_size, L, d)
        Ec = Ec.clone().transpose(0, 1);
        slide_len = Ec.shape[0];
        batch_size = Ec.shape[1];
        #output:(slide_len, batch_size, num_kernels, 1)
        output = torch.zeros((slide_len, batch_size, self.num_kernels, 1)).to(glb_var.get_value('device'));
        for idx, layer in enumerate(self.convlayers):
            ptr = idx%self.L;
            output[:, :, idx, :] = self.relu(self.maxpool_layers[ptr](layer(Ec)))[:, :, 0, :];
        #returned output:(batch_size, slide_len, num_kernels, 1) -> (batch_size, slide_len, 1, num_kernels)
        return output.transpose(0, 1).transpose(-1, -2);

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
    
    def forward(self, conv_output, attn, Eu):
        #conv_output:(batch_size, slide_len, 1, n)
        #attn:(batch_size, slide_len, 1, L*d)
        #fc1_input:(batch_size, slide_len, 1, n + L*d)
        fc1_input = torch.cat((conv_output, attn), dim = -1);
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

        #fc2_intput: (batch_size, slide_len, 2 * d) -> (batch_size, 2 * d)
        fc2_intput = torch.cat((Eu.repeat(1, slide_len, 1), u_attn), dim = -1).mean(dim = 1);
        #fc2_output: (batch_size, req_set_len) -> (batch_size, req_set_len)
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
        self.VrtConv = VrtConv(self.d, batch_size, self.n_kernels, self.L);
        self.self_attn = AttnNet(self.d);
        self.LSTFcNet = LSTFcNet(self.d, self.input_types, self.n_kernels, self.L);
    
    def forward(self, su):
        #su: (batch_size, req_len)
        #Ec: (batch_size, slide_len, L, d)
        Ec = self.encoder(su.unfold(-1, self.L, self.L));
        #Eu: (batch_size, 1, d)
        Eu = self.encoder(su).mean(dim = 1).unsqueeze(1);
        #o: (batch_size, slide_len, 1, n)
        o = self.VrtConv(Ec);
        #attn: (batch_size, slide_len, 1, L*d)
        attn = self.self_attn(Ec);
        #pro_logits: (batch_size, slide_len, T, req_set_len)
        pro_logits = self.LSTFcNet(o, attn, Eu);
        return pro_logits
    
    def cal_loss(self, su, next_item):
        '''Calculate loss for PSAC_gen
        '''
        #logits: (1, slide_len, T, req_types)
        logits = self.forward(su);
        return torch.nn.CrossEntropyLoss(ignore_index = self.mask_item[0])(logits.reshape(-1, logits.shape[-1]), next_item);