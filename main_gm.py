import sys
import time
import wandb
import pickle
import logging
import datetime
import argparse
import torch.nn as nn
from torch.backends import cudnn
from torch.utils.data import DataLoader, TensorDataset

from utils import util
from utils.util import *
from Trainer_nc import Trainer
from model.mlp import MLP
from model.loss import SupConLoss
from utils.measure_nc import analysis_feat

from imbalance_data.gm_data import make_blobs
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split


#data generation code
config_dt = dict(
n_samples = 300,  # Total number of samples
n_features = 3,   # Number of features (dimensionality)
n_cls = 3,        # Number of classes (Gaussian components)
n_sub_cls = 3,
cls_dist=[5, 8], sub_cls_dist=[2,4], sub_cls_std=1.5, #coarse distance between [] /subclass/subclass standard deviation
random_state = 42
)

# cls_dist=[5, 8], sub_cls_dist=[2,4], sub_cls_std=1.5 

# Merge config and argparse arguments
def update_args_with_dict(args, config):
    args_dict = vars(args)  # Convert argparse.Namespace to a dictionary
    for key, value in config.items():
        if key not in args_dict or args_dict[key] is None:
            setattr(args, key, value)
    return args


def train_one_epoch(model, criterion, optimizer, data_loader):
    model.train()
    losses = AverageMeter('Loss', ':.4e')
    train_acc = AverageMeter('Train_acc', ':.4e')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i, (inputs, labels) in enumerate(data_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        output, h = model(inputs, ret='of')
        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item())
        train_acc.update((output.argmax(dim=-1) == labels).float().mean().item(), len(labels))

    return losses.avg, train_acc.avg


def main(args):
    #random generated
    #save the generated data to make sure every epoch has the same data
    args = update_args_with_dict(args, config_dt)
    if args.coarse.startswith('f'):
        args.num_classes = args.n_cls * args.n_sub_cls
    elif args.coarse.startswith('c'):
        args.num_classes = args.n_cls
    args.store_name = f'{args.dataset}3_Y{args.coarse}_LR{args.lr}'
    print(args)

    # ============ Generate data from a mixture of Gaussians ============
    if args.new_data:
        x, fine_y, coarse_y = make_blobs(
            n_samples=args.n_samples, n_features=args.n_features,
            n_cls=args.n_cls, n_sub_cls=args.n_sub_cls,
            cls_dist=args.cls_dist, sub_cls_dist=args.sub_cls_dist, sub_cls_std=args.sub_cls_std,
            center_box=(-10.0, 10.0), shuffle=False, random_state=None
        )

        with open('gm_dt3.pkl', 'wb') as f:
            pickle.dump([x, fine_y, coarse_y], f)
    else:
        with open('gm_dt3.pkl', 'rb') as f:
            x, fine_y, coarse_y = pickle.load(f)

    #  ============ plot the data ============
    markers = ['*', 'o', '+']
    colors = ['C1', 'C2', 'C3']
    if args.n_features == 2:
        for cs_cls_id in range(args.n_cls):
            x_, fine_y_, coarse_y_ = x[coarse_y == cs_cls_id], fine_y[coarse_y == cs_cls_id], coarse_y[
                coarse_y == cs_cls_id]
            for counter, fn_cls_id in enumerate(np.unique(fine_y_)):
                plt.scatter(x_[fine_y_ == fn_cls_id, 0], x_[fine_y_ == fn_cls_id, 1], marker=markers[counter],
                            c=colors[cs_cls_id])

    if args.n_features == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for cs_cls_id in range(args.n_cls):
            x_, fine_y_, coarse_y_ = x[coarse_y == cs_cls_id], fine_y[coarse_y == cs_cls_id], coarse_y[
                coarse_y == cs_cls_id]
            for counter, fn_cls_id in enumerate(np.unique(fine_y_)):
                ax.scatter(x_[fine_y_ == fn_cls_id, 0], x_[fine_y_ == fn_cls_id, 1], x_[fine_y_ == fn_cls_id, 2],
                           marker=markers[counter], c=colors[cs_cls_id])

    # ============ split the data to train and test ============
    x_train, x_test, fy_train, fy_test, cy_train, cy_test = train_test_split(x, fine_y, coarse_y, test_size=0.3,
                                                                             stratify=fine_y, random_state=42)
    y_train = fy_train if args.coarse.startswith('f') else cy_train
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train)

    y_test = fy_test if args.coarse[1] == 'f' else cy_test
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test)

    train_set = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = True

    os.environ["WANDB_API_KEY"] = "cd3fbdd397ddb5a83b1235d177f4d81ce1200dbb"
    os.environ["WANDB_MODE"] = "online" #"dryrun"
    wandb.login(key='cd3fbdd397ddb5a83b1235d177f4d81ce1200dbb')
    wandb.init(project="gm",name=args.store_name)
    wandb.config.update(args)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ==================== create model
    model = MLP(in_dim=args.n_features, out_dim=args.num_classes, args=args, arch=args.arch)
    model = model.to(device)
    _ = print_model_param_nums(model=model)

    # ================= setup training
    if args.loss == 'ce':
        criterion = nn.CrossEntropyLoss(reduction='mean')
    elif args.loss == 'scon':
        criterion = SupConLoss(temperature=args.temp)
    optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ================= start training
    for epoch in range(args.epochs):

        loss, train_acc = train_one_epoch(model, criterion, optimizer, train_loader)
        lr_scheduler.step()

        if epoch == 0 or (epoch+1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                outputs, feats = model(x_train.to(device), ret='of')
            train_nc = analysis_feat(y_train, feats, args, W=model.fc.weight.data)

            with torch.no_grad():
                outputs, feats = model(x_test.to(device), ret='of')
            pred_test = outputs.argmax(dim=-1)
            if args.coarse == 'fc':
                pred_test = pred_test // args.n_cls
            test_acc = (pred_test == y_test.to(device)).float().mean()
            test_nc = analysis_feat(fy_test if args.coarse.startswith('f') else cy_test, feats, args)

            log_dt = {
                'train/train_loss': loss,
                'train/lr': optimizer.param_groups[0]['lr'],

                'train_nc/nc1': train_nc['nc1'],
                'train_nc/nc2': train_nc['nc2'],
                'train_nc/nc2h': train_nc['nc2h'],
                'train_nc/nc2w': train_nc['nc2w'],
                'train_nc/h_norm': train_nc['h_norm'],
                'train_nc/w_norm': train_nc['w_norm'],

                'val_nc/nc1': test_nc['nc1'],
                'val_nc/nc2h': test_nc['nc2h'],
            }

            log_dt.update({'train/acc_fine': train_acc}) if args.coarse[0] == 'f' else log_dt.update({'train/acc_coarse': train_acc})
            log_dt.update({'val/acc_fine': test_acc}) if args.coarse[1] == 'f' else log_dt.update({'val/acc_coarse': test_acc})

            wandb.log(log_dt, step=epoch)
            print(f"epoch:{epoch}, train loss:{loss:.4f}, train acc: {train_acc:.4f}, test acc: {test_acc:.4f}")


if __name__ == '__main__':
    # train set
    parser = argparse.ArgumentParser(description="Global and Local Mixture Consistency Cumulative Learning")
    parser.add_argument('--coarse', default='ff', type=str, help='f:False, t:Test at coarse level, b: Both train and test')
    parser.add_argument('--dataset', default='gm', type=str)
    parser.add_argument('--new_data', default=False, action='store_true')

    # model structure
    parser.add_argument('-a', '--arch', metavar='ARCH', default='mlp64_64_64')
    parser.add_argument('--num_classes', default=10, type=int, help='number of classes ')

    parser.add_argument('--loss', type=str, default='ce')  # ce|ls|ceh|hinge
    parser.add_argument('--temp', type=float, default=0.07)  # temperature for SupCon loss

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', '--learning-rate', default=0.05, type=float, metavar='LR', dest='lr')
    parser.add_argument('--scheduler', type=str, default='ms')
    parser.add_argument('--lr_decay', type=float, default=0.5)

    parser.add_argument('--epochs', default=500, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight_decay', default=5e-4, type=float, metavar='W', dest='weight_decay')

    # MLP settings (only when using mlp and res_adapt(in which case only width has effect))
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--bias', type=str, default='t')

    # etc.
    parser.add_argument('--seed', default=2021, type=int, help='seed for initializing training. ')
    args = parser.parse_args()

    main(args)