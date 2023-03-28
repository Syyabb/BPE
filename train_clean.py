import config
import torch
from classifier_models import PreActResNet18, ResNet18
from networks.models import NetC_MNIST
from utils.dataloader import PostTensorTransform, get_dataloader, get_dataset
from utils.utils import progress_bar

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
    criterion_CE = torch.nn.CrossEntropyLoss()

    transforms = PostTensorTransform(opt).to(opt.device)

    for batch_idx, (inputs, targets) in enumerate(train_dl):
        optimizerC.zero_grad()

        total_inputs, total_targets = inputs.to(opt.device), targets.to(opt.device)
        total_inputs = transforms(total_inputs)
        total_preds = netC(total_inputs)
        loss_ce = criterion_CE(total_preds, total_targets)

        loss = loss_ce
        loss.backward()

        optimizerC.step()

    schedulerC.step()


def eval(netC, test_dl, opt,):
    print(" Eval:")
    netC.eval()

    total_sample = 0
    total_clean_correct = 0

    for batch_idx, (inputs, targets) in enumerate(test_dl):
        with torch.no_grad():
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
            bs = inputs.shape[0]
            total_sample += bs

            # Evaluate Clean
            preds_clean = netC(inputs)
            total_clean_correct += torch.sum(torch.argmax(preds_clean, 1) == targets)

            acc_clean = total_clean_correct * 100.0 / total_sample


            info_string = "Clean Acc: {:.4f} ".format(
                acc_clean
            )
            progress_bar(batch_idx, len(test_dl), info_string)



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

    # Dataset
    train_dl = get_dataloader(opt, True)
    test_dl = get_dataloader(opt, False)

    # prepare model
    netC, optimizerC, schedulerC = get_model(opt)

    best_clean_acc = 0, 0
    for epoch in range(opt.n_iters):
        print("Epoch {}:".format(epoch + 1))
        train(netC, optimizerC, schedulerC, train_dl, opt)
        eval(
            netC,
            test_dl,
            opt,
        )
        torch.save(netC.state_dict(), './pt/clean_model.pt')


if __name__ == "__main__":
    main()