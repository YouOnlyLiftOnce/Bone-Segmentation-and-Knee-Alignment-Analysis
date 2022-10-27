import argparse
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import LonglegsDataset
from model import UNet
from train import train, resume, evaluate

'''
The dashboard of model training and testing
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    # training parameters
    parser.add_argument('--exp_id', type=str, default='exp_1_128x1024_dc_b=8')
    parser.add_argument('--resume', type=int, default=0, help='resume the trained model')
    parser.add_argument('--test', type=int, default=1, help='test with trained model')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-3, help='learning rate')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio of examples in validation set')
    parser.add_argument('--seed', type=int, default=1, help='random seed')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # load train and validation set
    trainvalset = LonglegsDataset(split="train")
    # split the trainvalset
    n = len(trainvalset)
    n_val = int(args.val_ratio * n)
    n_train = n - n_val
    trainset, valset = torch.utils.data.random_split(trainvalset, [n_train, n_val])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                            shuffle=False)

    testset = LonglegsDataset(split="test")
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False)

    dataloaders = (trainloader, valloader)
    print(f"dataset size, train set:{n_train}, val set:{n_val}, test set:{len(testset)}")

    # network
    model = UNet().to(device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=5)

    # resume the trained model
    if args.resume:
        model, optimizer = resume(args, model, optimizer)

    if args.test == 1:  # test mode, resume the trained model and test
        model_path = "exp_1_128x1024_dc_b=8_model.pth"
        model = torch.load(model_path)
        test_loss, dice, iou = evaluate(model, testloader)
        print('testing finished, dice coefficient: {:.3f}, IoU {:.3f}: '.format(dice, iou), )
    else:  # train mode, train the network from scratch
        train(args, model, optimizer, dataloaders, scheduler)
        print('training finished')
