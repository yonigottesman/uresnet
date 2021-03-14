import argparse
import time

import torch as torch
from tqdm import tqdm

from config import Config
from data import gen_datasets, get_dataloader
from loss import FeatureLoss
from model import ResnetUnet


class AverageMeter(object):
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def single_epoch(dataloader,
                 model,
                 loss_function,
                 optimizer=None,
                 scheduler=None):

    losses = AverageMeter()

    for x, y in tqdm(dataloader, leave=False):
        y_hat = model(x.to(Config.DEVICE))
        y = y.to(Config.DEVICE)
        loss = loss_function(y_hat, y)

        if model.training:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        batch_size = len(y)
        losses.update(loss.item(), batch_size)
    return losses.avg


def fit(epochs, model, train_dl, valid_dl, loss_func, optimizer, scheduler,
        save_path):

    min_loss = 10
    for epoch in range(epochs):
        start_time = time.time()

        lr = optimizer.param_groups[0]['lr']
        model.train()
        train_loss = single_epoch(train_dl, model, loss_func, optimizer,
                                  scheduler)

        model.eval()
        with torch.no_grad():
            valid_loss = single_epoch(valid_dl, model, loss_func)

        if valid_loss < min_loss:
            min_loss = valid_loss
            torch.save(model, save_path)

        scheduler.step(valid_loss)

        secs = int(time.time() - start_time)
        print(f'Epoch {epoch} {secs}[sec] lr={lr:.6f}', end=' ')
        print(f'Train: loss {train_loss:.4f}. ', end='\t')
        print(f'Valid: loss {valid_loss:.4f}.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='.cache', type=str)
    parser.add_argument('--resnet', default='resnet101', type=str)
    parser.add_argument('--output_model_path',
                        default=f'colornet.pt',
                        type=str)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--wd', default=0.0, type=float)
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--bs', default=16, type=int)

    args = parser.parse_args()

    model = ResnetUnet(args.resnet).to(Config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.wd)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           'min',
                                                           patience=2)

    train_dl, val_dl = get_dataloader((256, 256), bs=args.bs)

    fit(args.epochs, model, train_dl, val_dl, FeatureLoss([20, 70, 10]),
        optimizer, scheduler, args.output_model_path)
