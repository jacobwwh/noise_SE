import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Loss functions
def loss_coteaching(y_1, y_2, t, true_t, forget_rate, ind=None, noise_or_not=None):
    loss_1 = F.cross_entropy(y_1, t, reduction='none')
    ind_1_sorted = np.argsort(loss_1.data.cpu()).cuda()
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = F.cross_entropy(y_2, t, reduction='none')
    ind_2_sorted = np.argsort(loss_2.data.cpu()).cuda()
    loss_2_sorted = loss_2[ind_2_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))

    batch_noise_or_not=np.array(t.cpu()==true_t.cpu())  #clean->true, noisy->false
    pure_ratio_1=np.sum(batch_noise_or_not[ind_1_sorted[:num_remember].cpu()])/float(num_remember) #ratio of correct and remembered samples in a batch
    pure_ratio_2=np.sum(batch_noise_or_not[ind_2_sorted[:num_remember].cpu()])/float(num_remember)

    ind_1_update=ind_1_sorted[:num_remember]
    ind_2_update=ind_2_sorted[:num_remember]
    # exchange
    loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])

    return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember, pure_ratio_1, pure_ratio_2
