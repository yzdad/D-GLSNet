import torch
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, ExponentialLR


def build_optimizer(model, config):
    if config['name'] == "adam":
        return torch.optim.Adam(model.parameters(), lr=config['ture_lr'], weight_decay=config['adam_decay'])
    elif config['name'] == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=config['ture_lr'], weight_decay=config['adamw_decay'])
    else:
        raise ValueError(f"TRAINER.OPTIMIZER = {config['name']} is not a valid optimizer!")


def build_scheduler(config, optimizer):
    """
    Returns:
        scheduler (dict):{
            'scheduler': lr_scheduler,
            'interval': 'step',  # or 'epoch'
            'monitor': 'val_f1', (optional)
            'frequency': x, (optional)
        }
    """
    scheduler = {'interval': config['scheduler_interval']}

    if config['scheduler'] == 'MultiStepLR':
        scheduler.update({'scheduler': MultiStepLR(optimizer, config['mslr_milestones'], gamma=config['mslr_gamma'])})
    elif config['scheduler'] == 'CosineAnnealing':
        scheduler.update({'scheduler': CosineAnnealingLR(optimizer, config['cosa_tmax'])})
    elif config['scheduler'] == 'ExponentialLR':
        scheduler.update({'scheduler': ExponentialLR(optimizer, config['elr_camma'])})
    else:
        raise NotImplementedError()

    return scheduler
