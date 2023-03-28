import json
import os
import shutil
from time import time
import config
import torch
import torch.nn.functional as F
from classifier_models import PreActResNet18, ResNet18
from networks.models import NetC_MNIST
from torch.utils.tensorboard import SummaryWriter
from utils.dataloader import PostTensorTransform
from utils.utils import progress_bar
from utils.dataloader import get_dataset, get_dataloader, Customer_dataset_warp
import torchvision.transforms as transforms


def get_model(opt):
    netC = None
    optimizerC = None
    schedulerC = None

    if opt.dataset == "cifar10" or opt.dataset == "gtsrb":
        netC = PreActResNet18(num_classes=opt.num_classes).to(opt.device)
    if opt.dataset == "celeba":
        netC = ResNet18().to(opt.device)
    if opt.dataset == "mnist":
        netC = NetC_MNIST().to(opt.device)

    # Optimizer
    optimizerC = torch.optim.SGD(netC.parameters(), opt.lr_C, momentum=0.9, weight_decay=5e-4)

    # Scheduler
    schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, opt.schedulerC_milestones, opt.schedulerC_lambda)

    return netC, optimizerC, schedulerC


def train(netC, optimizerC, schedulerC, train_dl, opt):
    print(" Train:")
    netC.train()
    total_loss_ce = 0
    total_sample = 0

    total_clean_correct = 0
    criterion_CE = torch.nn.CrossEntropyLoss()
    transforms = PostTensorTransform(opt).to(opt.device)
    total_time = 0

    for batch_idx, (inputs, targets) in enumerate(train_dl):
        optimizerC.zero_grad()

        inputs, total_targets = inputs.to(opt.device), targets.to(opt.device)
        bs = len(inputs)
        total_inputs = transforms(inputs)

        start = time()
        total_preds = netC(total_inputs)
        total_time += time() - start

        loss_ce = criterion_CE(total_preds, total_targets)

        loss = loss_ce
        loss.backward()

        optimizerC.step()

        total_sample += bs
        total_loss_ce += loss_ce.detach()

        total_clean_correct += torch.sum(
            torch.argmax(total_preds, dim=1) == total_targets
        )

    avg_acc_clean = total_clean_correct * 100.0 / total_sample
    avg_loss_ce = total_loss_ce * 100.0 / total_sample

    print("CE Loss: {:.4f} | Clean Acc: {:.4f} ".format(avg_loss_ce, avg_acc_clean))

    schedulerC.step()


def eval(
    netC,
    test_dl,
    noise_grid,
    identity_grid,
    opt,
):
    print(" Eval:")
    netC.eval()

    total_sample = 0
    total_clean_correct = 0
    total_bd_correct = 0
    total_cross_correct = 0

    for batch_idx, (inputs, targets) in enumerate(test_dl):
        with torch.no_grad():
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
            bs = inputs.shape[0]
            total_sample += bs

            # Evaluate Clean
            preds_clean = netC(inputs)
            total_clean_correct += torch.sum(torch.argmax(preds_clean, 1) == targets)

            # Evaluate Backdoor
            grid_temps = (identity_grid + opt.s * noise_grid / opt.input_height) * opt.grid_rescale
            grid_temps = torch.clamp(grid_temps, -1, 1)

            ins = torch.rand(bs, opt.input_height, opt.input_height, 2).to(opt.device) * 2 - 1
            grid_temps2 = grid_temps.repeat(bs, 1, 1, 1) + ins / opt.input_height
            grid_temps2 = torch.clamp(grid_temps2, -1, 1)

            inputs_bd = F.grid_sample(inputs, grid_temps.repeat(bs, 1, 1, 1), align_corners=True)
            if opt.attack_mode == "all2one":
                targets_bd = torch.ones_like(targets) * opt.target_label
            if opt.attack_mode == "all2all":
                targets_bd = torch.remainder(targets+1, opt.num_classes)
            preds_bd = netC(inputs_bd)
            total_bd_correct += torch.sum(torch.argmax(preds_bd, 1) == targets_bd)

            if opt.cross_ratio:
                inputs_cross = F.grid_sample(inputs, grid_temps2, align_corners=True)
                preds_cross = netC(inputs_cross)
                total_cross_correct += torch.sum(torch.argmax(preds_cross, 1) == targets)


    acc_clean = total_clean_correct * 100.0 / total_sample
    acc_bd = total_bd_correct * 100.0 / total_sample
    acc_cross = total_cross_correct * 100.0 / total_sample

    info_string = "Clean Acc: {:.4f}| Bd Acc: {:.4f} ".format(
        acc_clean, acc_bd
    )
    print(info_string)



def main():
    opt = config.get_arguments().parse_args()

    if opt.dataset in ["mnist", "cifar10"]:
        opt.num_classes = 10
    elif opt.dataset == "gtsrb":
        opt.num_classes = 43
    elif opt.dataset == "celeba":
        opt.num_classes = 8
    else:
        raise Exception("Invalid Dataset")

    if opt.dataset == "cifar10":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "gtsrb":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "mnist":
        opt.input_height = 28
        opt.input_width = 28
        opt.input_channel = 1
    elif opt.dataset == "celeba":
        opt.input_height = 64
        opt.input_width = 64
        opt.input_channel = 3
    else:
        raise Exception("Invalid Dataset")

    # prepare model
    netC, optimizerC, schedulerC = get_model(opt)


    # Prepare grid
    ins = torch.rand(1, 2, opt.k, opt.k) * 2 - 1
    ins = ins / torch.mean(torch.abs(ins))
    noise_grid = (
        F.upsample(ins, size=opt.input_height, mode="bicubic", align_corners=True)
        .permute(0, 2, 3, 1)
        .to(opt.device)
    )
    array1d = torch.linspace(-1, 1, steps=opt.input_height)
    x, y = torch.meshgrid(array1d, array1d)
    identity_grid = torch.stack((y, x), 2)[None, ...].to(opt.device)

    '''
    state_dict = torch.load("./pt/cifar10_all2one_morph.pth.tar")
    netC.load_state_dict(state_dict["netC"])
    optimizerC.load_state_dict(state_dict["optimizerC"])
    schedulerC.load_state_dict(state_dict["schedulerC"])
    '''

    dataset = get_dataset(opt, True)
    t = transforms.Compose([transforms.ToTensor()])
    dataset = Customer_dataset_warp(dataset, t, opt, noise_grid, identity_grid)
    train_dl = torch.utils.data.DataLoader(dataset, batch_size=opt.bs, num_workers=opt.num_workers, shuffle=True)
    test_dl = get_dataloader(opt, False)

    for epoch in range(opt.n_iters):
        print("Epoch {}:".format(epoch + 1))
        train(netC, optimizerC, schedulerC, train_dl, opt)
        eval(
            netC,
            test_dl,
            noise_grid,
            identity_grid,
            opt,
        )


if __name__ == "__main__":
    main()
