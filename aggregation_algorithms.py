import torch
#device=torch.device("cuda:1")

def FedAvg(state_dict,n_devices,dataset_len,device):
    updated_dict=state_dict[0].copy()
    for layer in state_dict[0].keys():
        averaged_params=torch.zeros(state_dict[0][layer].size(),device=device)
        for dev in range(n_devices):
            averaged_params+=dataset_len[dev]*state_dict[dev][layer]
        updated_dict[layer]=(averaged_params/dataset_len.sum()) #sum (nk/n)*w^(k)(t+1)
    return updated_dict
