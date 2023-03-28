import torch
import os
import torch.nn as nn
import copy
import torch.nn.functional as F
from config import get_arguments
import numpy as np

import sys

sys.path.insert(0, "../..")
from utils.dataloader import get_dataloader
from utils.utils import progress_bar
from classifier_models import PreActResNet18


def create_targets_bd(targets, opt):
    if opt.attack_mode == "all2one":
        bd_targets = torch.ones_like(targets) * opt.target_label
    elif opt.attack_mode == "all2all":
        bd_targets = torch.tensor([(label + 1) % opt.num_classes for label in targets])
    else:
        raise Exception("{} attack mode is not implemented".format(opt.attack_mode))
    return bd_targets.to(opt.device)

#读取文件夹下所有图像作为Blend Set
blends_path = './../../hellokity'
blends_path = [os.path.join(blends_path, i) for i in os.listdir(blends_path)]
from PIL import Image
import torchvision.transforms as transforms
# t = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])])
t = transforms.Compose([transforms.ToTensor()])
blends_imgs = [t(Image.open(i).convert('RGB').resize((32, 32))).unsqueeze(0).cuda() for i in blends_path]

r = 0.1
def create_bd(inputs, targets, opt):
    bd_targets = create_targets_bd(targets, opt)
    blend_indexs = np.random.randint(0, len(blends_imgs), (5,))
    bd_inputs = inputs * (1-r) + blends_imgs[blend_indexs[0]] * r
    return bd_inputs, bd_targets

def eval(netC, test_dl, best_clean_acc, best_bd_acc, opt,):
    print(" Eval:")
    netC.eval()

    total_sample = 0
    total_clean_correct = 0
    total_bd_correct = 0


    for batch_idx, (inputs, targets) in enumerate(test_dl):
        with torch.no_grad():
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
            bs = inputs.shape[0]
            total_sample += bs

            # Evaluate Clean
            preds_clean = netC(inputs)
            total_clean_correct += torch.sum(torch.argmax(preds_clean, 1) == targets)

            # Evaluate Backdoor
            with torch.no_grad():
                bs = inputs.shape[0]
                inputs_bd, targets_bd = create_bd(inputs, targets, opt)

            preds_bd = netC(inputs_bd)
            total_bd_correct += torch.sum(torch.argmax(preds_bd, 1) == targets_bd)

            acc_clean = total_clean_correct * 100.0 / total_sample
            acc_bd = total_bd_correct * 100.0 / total_sample


            info_string = "Clean Acc: {:.4f} - Best: {:.4f} | Bd Acc: {:.4f} - Best: {:.4f}".format(
                acc_clean, best_clean_acc, acc_bd, best_bd_acc
            )
            progress_bar(batch_idx, len(test_dl), info_string)
    # print(acc_clean, acc_bd)
    return best_clean_acc, best_bd_acc



def main():
    # Prepare arguments
    opt = get_arguments().parse_args()
    if opt.dataset == "cifar10":
        opt.num_classes = 10
    elif opt.dataset == "gtsrb":
        opt.num_classes = 43
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
    else:
        raise Exception("Invalid Dataset")

    # Load models
    if opt.dataset == "cifar10":
        netC = PreActResNet18().to(opt.device)
    elif opt.dataset == "gtsrb":
        netC = PreActResNet18(num_classes=43).to(opt.device)
    else:
        raise Exception("Invalid dataset")


    # netC.load_state_dict(torch.load('/data1/bingxu/pycharm_ssh/backdoor_sota/warp_and_blend_uap/blend_no_norm3.pt'))
    netC.load_state_dict(torch.load('/data1/bingxu/pycharm_ssh/backdoor_sota/warp_and_blend_uap/all2all_blend.pt'))
    # netC.load_state_dict(torch.load('/data1/bingxu/pycharm_ssh/backdoor_sota/gtsrb_warp/blend_iter.pt'))
    netC.to(opt.device)
    netC.eval()
    netC.requires_grad_(False)

    test_dl = get_dataloader(opt, train=False)

    # Forward hook for getting layer's output
    container = []

    def forward_hook(module, input, output):
        container.append(output)

    hook = netC.layer4.register_forward_hook(forward_hook)

    # Forwarding all the validation set
    print("Forwarding all the validation dataset:")
    for batch_idx, (inputs, _) in enumerate(test_dl):
        inputs = inputs.to(opt.device)
        netC(inputs)
        progress_bar(batch_idx, len(test_dl))

    # Processing to get the "more important mask"
    container = torch.cat(container, dim=0)
    activation = torch.mean(container, dim=[0, 2, 3])
    seq_sort = torch.argsort(activation)
    pruning_mask = torch.ones(seq_sort.shape[0], dtype=bool)
    hook.remove()

    # Pruning times - no-tuning after pruning a channel!!!
    acc_clean = []
    acc_bd = []
    opt.outfile = "{}_results.txt".format(opt.dataset)
    with open(opt.outfile, "w") as outs:
        for index in range(pruning_mask.shape[0]):
            net_pruned = copy.deepcopy(netC)
            num_pruned = index
            if index:
                channel = seq_sort[index - 1]
                pruning_mask[channel] = False
            print("Pruned {} filters".format(num_pruned))

            net_pruned.layer4[1].conv2 = nn.Conv2d(
                pruning_mask.shape[0], pruning_mask.shape[0] - num_pruned, (3, 3), stride=1, padding=1, bias=False
            )
            net_pruned.linear = nn.Linear(pruning_mask.shape[0] - num_pruned, 10)

            # Re-assigning weight to the pruned net
            for name, module in net_pruned._modules.items():
                if "layer4" in name:
                    module[1].conv2.weight.data = netC.layer4[1].conv2.weight.data[pruning_mask]
                    module[1].ind = pruning_mask
                elif "linear" == name:
                    module.weight.data = netC.linear.weight.data[:, pruning_mask]
                    module.bias.data = netC.linear.bias.data
                else:
                    continue
            net_pruned.to(opt.device)
            if index % 10 == 0:
                clean, bd = eval(net_pruned, test_dl, 0, 0, opt)
            outs.write("%d %0.4f %0.4f\n" % (index, clean, bd))


if __name__ == "__main__":
    main()
