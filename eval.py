import torch
import torch.nn.functional as F
from tqdm import tqdm


def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    PET_type = torch.float32
    n_val = len(loader)  # the number of batch
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            CTs, true_PETs = batch['CT'], batch['PET']
            CTs = CTs.to(device=device, dtype=torch.float32)
            true_PETs = true_PETs.to(device=device, dtype=PET_type)

            with torch.no_grad():
                PET_pred = net(CTs)

            tot += F.mse_loss(PET_pred, true_PETs, reduction='mean').item()
            pbar.update()
            
        tot = tot/n_val
        
    net.train()
    return tot
