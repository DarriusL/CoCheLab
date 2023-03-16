import torch

class BPRLoss(torch.nn.Module):
    """ BPRLoss, based on Bayesian Personalized Ranking

    Parameters:
    -----------
    gamma:float, optional 
    eps to avoid  nan
    default:1e-10
    """

    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        '''
        pos_score and neg_score in the same shape
        '''
        loss = -torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss