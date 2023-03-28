import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import config
import torch
from classifier_models import PreActResNet18, ResNet18
from networks.models import NetC_MNIST
from utils.dataloader import get_dataloader
from aft_train import aft_train, eval, eval_warp
from torch.utils.tensorboard import SummaryWriter

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

    netC, optimizerC, schedulerC = get_model(opt)


    log_path = os.path.join('./log', opt.trigger_type)
    writer = SummaryWriter(log_path)
    train_dl = get_dataloader(opt, train=True)
    test_dl = get_dataloader(opt, False)

    if opt.trigger_type == 'warp':
        state_dict = torch.load(opt.model_path)
        netC.load_state_dict(state_dict["netC"])
        eval_warp(netC, test_dl, opt, writer, 0)
    else:
        netC.load_state_dict(torch.load(opt.model_path))
        eval(netC, test_dl, opt, writer, 0)



    for epoch in range(20):
        print("Epoch {}:".format(epoch + 1))
        if epoch <= 2:
            aft_train(netC, optimizerC, schedulerC, train_dl, opt, adv=True)
        else:
            aft_train(netC, optimizerC, schedulerC, train_dl, opt, adv=False)

        if (epoch + 1) % 1 == 0:
            if opt.trigger_type == 'warp':
                eval_warp(netC, test_dl, opt, writer, epoch + 1)
            else:
                eval(netC, test_dl, opt, writer, epoch + 1)



if __name__ == "__main__":
    main()
