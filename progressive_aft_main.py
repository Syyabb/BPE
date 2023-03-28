import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from torch.utils.data import Dataset
import config
import numpy as np
import torch
import torchvision.transforms as transforms
from classifier_models import PreActResNet18, ResNet18
from networks.models import NetC_MNIST
from utils.dataloader import get_dataloader, Customer_dataset, get_dataset
from utils.utils import get_cos_similar
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

    netC.eval()

    dataset = get_dataset(opt, train=True)
    trigger_index = np.random.permutation(len(dataset))[0: int(len(dataset) * opt.pc)]
    t = transforms.Compose([transforms.ToTensor()])
    dataset = Customer_dataset(opt, dataset, transform=t, trigger_index=trigger_index)
    train_dl = torch.utils.data.DataLoader(dataset, batch_size=opt.bs, num_workers=opt.num_workers, shuffle=False)

    log_path = os.path.join('./log', opt.trigger_type)
    writer = SummaryWriter(log_path)
    test_dl = get_dataloader(opt, False)

    if opt.trigger_type == 'warp':
        state_dict = torch.load(opt.model_path)
        netC.load_state_dict(state_dict["netC"])
        eval_warp(netC, test_dl, opt, writer, 0)
    else:
        netC.load_state_dict(torch.load(opt.model_path))
        eval(netC, test_dl, opt, writer, 0)

    netC.eval()
    labels_backdoor = []
    for batch_idx, (inputs, targets) in enumerate(train_dl):
        with torch.no_grad():
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
            preds_clean = netC(inputs).cpu().numpy()
            for i in range(len(preds_clean)):
                labels_backdoor.append(preds_clean[i])
    filter_index = None

    for progressive_step in range(4):
        #netC.load_state_dict(torch.load('./all2all_blend.pt'))

        if filter_index is None:
            filter_index = np.random.permutation(len(dataset))[0: int(len(dataset) * 0.05)]
        extra_clean_data_idx = [True] * len(train_dl.dataset)
        for i in filter_index:
            extra_clean_data_idx[i] = False
        dataset.filter(extra_clean_data_idx)

        top_rates = [0.2, 0.5, 0.7]

        if progressive_step == 0:
            idx_ex = filter_index
        else:
            idx_ex = idx[:int(len(idx) * top_rates[progressive_step - 1])]

        count = len(idx_ex)
        count_bd = 0
        for i in idx_ex:
            if i in trigger_index:
                count_bd += 1
        rate = count_bd / count

        print("this step for purify backdoor rate is {:.4f} , total clean dataset size is{}".format(rate, count))

        for epoch in range(10):
            print("Epoch {}:".format(epoch + 1))
            if progressive_step > 1:
                eps = 2 / 255.
            if progressive_step == 0:
                aft_train(netC, optimizerC, schedulerC, train_dl, opt, adv=True)
            # elif progressive_step == 1:
            #     train(netC, optimizerC, schedulerC, train_dl, opt, adv=True, partial=True)
            else:
                if epoch == 0:
                    aft_train(netC, optimizerC, schedulerC, train_dl, opt, adv=True, partial=True)
                else:
                    aft_train(netC, optimizerC, schedulerC, train_dl, opt, adv=False)

            if (epoch + 1) % 1 == 0:
                if opt.trigger_type == 'warp':
                    eval_warp(netC, test_dl, opt, writer, epoch + 1 + progressive_step * 10)
                else:
                    eval(netC, test_dl, opt, writer, epoch + 1 + progressive_step * 10)

        extra_clean_data_idx = [False] * len(dataset.full_dataset)
        dataset.filter(extra_clean_data_idx)

        netC.eval()
        labels_clean = []
        for batch_idx, (inputs, targets) in enumerate(train_dl):
            with torch.no_grad():
                inputs, targets = inputs.to(opt.device), targets.to(opt.device)
                preds_clean = netC(inputs).cpu().numpy()
                for i in range(len(preds_clean)):
                    labels_clean.append(preds_clean[i])

        labels_backdoor = np.array(labels_backdoor)
        labels_clean = np.array(labels_clean)

        a = []
        for i in range(len(labels_backdoor)):
            a.append(get_cos_similar(labels_backdoor[i], labels_clean[i]))
        a = np.array(a)
        idx = np.argsort(a)[::-1]
        # if progressive_step == 0:
        #     idx = np.argsort(a)
        # else:
        #     idx = np.argsort(a)[::-1]

        filter_index = idx[:int(len(idx) * top_rates[progressive_step])]
        print(len(filter_index))

if __name__ == "__main__":
    main()
