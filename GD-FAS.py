import argparse, json, os, random, torch, numpy
from utils import *
from models import build_model, build_optimizer
from data import build_datasets
import torch.optim as optim
from datetime import datetime

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

def log_f(f, console=True):
    def log(msg):
        with open(f, 'a') as file:
            file.write(msg)
            file.write('\n')
        if console:
            if tqdm is not None:
                tqdm.write(msg)
            else:
                print(msg)
    return log

def json_default(value):
    if isinstance(value, (numpy.integer, numpy.floating, numpy.bool_)):
        return value.item()
    if isinstance(value, numpy.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')

def append_jsonl(path, payload):
    with open(path, 'a', encoding='utf-8') as file:
        json.dump(payload, file, default=json_default)
        file.write('\n')

def write_json(path, payload):
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(payload, file, indent=2, default=json_default)
        file.write('\n')

def parse_args():
    parser = argparse.ArgumentParser()
    # build dirs
    parser.add_argument('--backbone', type=str, default="clip", help='backborn - resnet18 or clip')
    parser.add_argument('--silence', action='store_true')
    parser.add_argument('--log_name', type=str, default="test", help='log')
    parser.add_argument('--seed', type=int, default=2025, help='')
    parser.add_argument('--data_root', type=str, default="datasets", help='YOUR_Data_Dir')
    parser.add_argument('--protocol', type=str, default="O_C_I_to_M", help='O_C_I_to_M, O_M_I_to_C, O_C_M_to_I, I_C_M_to_O')
    parser.add_argument('--max_iter', type=int, default=400, help='')
    parser.add_argument('--batch_size', type=int, default=16, help='')
    parser.add_argument('--lr', type=float, default=0.000003, help='')
    parser.add_argument('--wd', type=float, default=0.000001, help='')
    parser.add_argument('--gs', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--beta', type=float, default=1.5, help='')
    parser.add_argument('--temperature', type=float, default=0.1, help='') 
    parser.add_argument('--params', nargs=4, type=float, default=[1.0, 0.8, 0.1, 1.0])
    parser.add_argument('--step_size', type=int, default=10, help='')
    parser.add_argument('--prompt_mode', type=str, default='fixed', choices=['fixed', 'coop'], help='text prompt mode')
    parser.add_argument('--n_ctx', type=int, default=4, help='number of shared CoOp context tokens')
    parser.add_argument('--coop_prompt_lr_ratio', type=float, default=1.0, help='learning-rate multiplier for CoOp prompt parameters')
    return parser.parse_args()

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):
    # print to txt file
    log_path = 'results/{}'.format(args.log_name)
    os.makedirs(log_path, exist_ok=True)
    text_log_path = os.path.join(log_path, '{}.txt'.format(args.protocol))
    metrics_log_path = os.path.join(log_path, '{}.metrics.jsonl'.format(args.protocol))
    summary_log_path = os.path.join(log_path, '{}.summary.json'.format(args.protocol))
    print = log_f(text_log_path)
    run_id = datetime.now().strftime('%Y%m%d-%H%M%S')
    config = {
        'run_id': run_id,
        'log_name': args.log_name,
        'protocol': args.protocol,
        'backbone': args.backbone,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': args.wd,
        'seed': args.seed,
        'max_iter': args.max_iter,
        'step_size': args.step_size,
        'gs': args.gs,
        'beta': args.beta,
        'temperature': args.temperature,
        'prompt_mode': args.prompt_mode,
        'coop_prompt_lr_ratio': args.coop_prompt_lr_ratio,
        'params': args.params,
        'save': args.save,
    }
    append_jsonl(metrics_log_path, {
        'event': 'run_started',
        'config': config,
    })

    # setup
    train_loader, test_loader = build_datasets(args)
    networks = build_model(args)
    optimizer = build_optimizer(args, networks)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    networks.cuda()

    # training
    if not args.silence:
        print('------------------------------------------------------')
        print('information')
        print('------------------------------------------------------')
        print(f'{"log name":20} : {args.log_name}')
        print(f'{"protocol name":20} : {args.protocol}')
        print(f'{"backbone":20} : {args.backbone}')
        print(f'{"batch size":20} : {args.batch_size}')
        print(f'{"learning rate":20} : {args.lr}')
        print(f'{"weight decay":20} : {args.wd}')
        print(f'{"seed":20} : {args.seed}')
        print(f'{"max iter":20} : {args.max_iter}')
        print(f'{"step size":20} : {args.step_size}')
        print(f'{"gs":20} : {args.gs}')
        print(f'{"beta":20} : {args.beta}')
        print(f'{"temperature":20} : {args.temperature}')
        print(f'{"prompt mode":20} : {args.prompt_mode}')
        print(f'{"n_ctx":20} : {args.n_ctx}')
        print(f'{"coop lr ratio":20} : {args.coop_prompt_lr_ratio}')
        print(f'{"parameter1":20} : {args.params[0]}')
        print(f'{"parameter2":20} : {args.params[1]}')
        print(f'{"parameter3":20} : {args.params[2]}')
        print(f'{"parameter4":20} : {args.params[3]}')
        print('------------------------------------------------------')
        print('training')
        print('------------------------------------------------------')

    best_select = {
        'hter': 1.0,
        'message': "",
        'metrics': None,
        'iter': None,
        'epoch': None,
        'checkpoint_path': None,
    }
    progress_bar = None
    if tqdm is not None:
        progress_bar = tqdm(
            train_loader,
            total=len(train_loader),
            disable=args.silence,
            dynamic_ncols=True,
            desc='training',
        )
    train_iterator = progress_bar if progress_bar is not None else train_loader

    for iter, batch_samples in enumerate(train_iterator):
        epoch = iter // 10
        networks.train()
        optimizer.zero_grad()

        image_x_v1 = torch.cat([batch_samples[key]['image_x_v1'] for key in batch_samples])
        image_x_v2 = torch.cat([batch_samples[key]['image_x_v2'] for key in batch_samples])

        images     = torch.cat([image_x_v1, image_x_v2]).cuda()
        labels     = torch.cat([batch_samples[key]['label'] for key in batch_samples]).repeat(2).cuda()
        domains    = torch.cat([batch_samples[key]['domain'] for key in batch_samples]).repeat(2).cuda()

        loss = networks.compute_loss(images, labels, domains)
        # break
        loss.backward()
        optimizer.step()

        if progress_bar is not None:
            progress_bar.set_postfix(loss=f'{loss.item():.4f}')

        if (iter % 10 == 0) & (iter!=0):
            scheduler.step()
            infos = networks.loss_reset()
            print(f'epoch : {epoch} {infos["loss"]}')
            print('------------------------------------------------------------------------------------------------------------')
            list_scores = {}
            networks.eval()
            with torch.no_grad():
                list_scores = []
                for test_batch_samples in test_loader:
                    images  = test_batch_samples['image_x'].cuda()
                    labels  = test_batch_samples['label'].cuda()
                    logits, _ = networks(images)
                    # probs1 = torch.nn.functional.softmax(similarity, dim=1)
                    probs = torch.nn.functional.softmax(logits, dim=1)
                    for prob, label in zip(probs, labels):
                        list_scores.append("{} {}\n".format(prob[1].item(),label.item()))
                    
                test_ACC, tpr_filtered_1p, HTER, auc_test, val_threshold, val_ece, val_acc, sc, la = eval(list_scores)
                print("ACC_val:{:.4f} HTER_val:{:.4f} AUC:{:.4f} fpr1p:{:.4f} ECE:{:.4f} acc:{:.4f} threshold:{:.4f} ".format(
                        test_ACC[0], HTER[0], auc_test, tpr_filtered_1p, val_ece, val_acc, val_threshold))

                current_metrics = {
                    'run_id': run_id,
                    'event': 'evaluation',
                    'iter': iter,
                    'epoch': epoch,
                    'train_loss': loss.item(),
                    'train_loss_text': infos['loss'],
                    'acc_threshold': test_ACC[0],
                    'acc_05': test_ACC[1],
                    'acc_p': test_ACC[2],
                    'hter_threshold': HTER[0],
                    'hter_05': HTER[1],
                    'hter_p': HTER[2],
                    'auc': auc_test,
                    'tpr_at_fpr_1_percent': tpr_filtered_1p,
                    'ece': val_ece,
                    'calibration_acc': val_acc,
                    'threshold': val_threshold,
                }

                is_best = best_select['hter'] >= HTER[0]
                current_metrics['is_best'] = is_best
                append_jsonl(metrics_log_path, current_metrics)

                if is_best:
                    best_select['hter'] = HTER[0]
                    best_select['message'] = "ACC_val:{:.4f} HTER_val:{:.4f} AUC:{:.4f} fpr1p:{:.4f} ECE:{:.4f} acc:{:.4f} threshold:{:.4f} ".format(
                        test_ACC[0], HTER[0], auc_test, tpr_filtered_1p, val_ece, val_acc, val_threshold)
                    best_select['metrics'] = current_metrics
                    best_select['iter'] = iter
                    best_select['epoch'] = epoch
                    if args.save:
                        checkpoint_path = f'results/{args.log_name}/{args.protocol}_best.pth'
                        torch.save(networks, checkpoint_path)
                        best_select['checkpoint_path'] = checkpoint_path

                if progress_bar is not None:
                    progress_bar.set_postfix(loss=f'{loss.item():.4f}', best_hter=f'{best_select["hter"]:.4f}')

                print(f'best_hter: {best_select["hter"]:.4f}')
                print('------------------------------------------------------------------------------------------------------------')
    if progress_bar is not None:
        progress_bar.close()

    summary = {
        'run_id': run_id,
        'config': config,
        'text_log_path': text_log_path,
        'metrics_log_path': metrics_log_path,
        'best': {
            'iter': best_select['iter'],
            'epoch': best_select['epoch'],
            'hter': best_select['hter'],
            'checkpoint_path': best_select['checkpoint_path'],
            'metrics': best_select['metrics'],
        },
    }
    write_json(summary_log_path, summary)
    append_jsonl(metrics_log_path, {
        'run_id': run_id,
        'event': 'run_finished',
        'summary_path': summary_log_path,
        'best_hter': best_select['hter'],
    })
    print(best_select['message'])

if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.seed)
    main(args)
