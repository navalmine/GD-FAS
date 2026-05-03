from models import networks
import torch

def backbone_map(backbone):
    if 'resnet' in backbone:
        return 'resnet'
    elif 'clip' in backbone:
        return 'clip_encoder'
    elif 'safas' in backbone:
        return 'resnet18'

def build_model(args):
    return getattr(networks, backbone_map(args.backbone))(args)

def build_optimizer(args, net):
    print()
    if 'clip' in args.backbone:
        if getattr(args, 'prompt_mode', 'fixed') == 'coop':
            prompt_params, base_params = net.get_coop_param_groups()
            param_groups = []
            if base_params:
                param_groups.append({
                    'params': base_params,
                    'lr': args.lr,
                    'weight_decay': args.wd,
                })
            if prompt_params:
                param_groups.append({
                    'params': prompt_params,
                    'lr': args.lr * args.coop_prompt_lr_ratio,
                    'weight_decay': args.wd,
                })
            return torch.optim.Adam(param_groups, lr=args.lr, weight_decay=args.wd)
        return torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        return torch.optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.wd)
