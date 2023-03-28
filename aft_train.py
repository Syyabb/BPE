import torch
from utils.dataloader import PostTensorTransform
from create_bd import patch, sig, blend, generate
import torch.nn.functional as F

def aft_train(netC, optimizerC, schedulerC, train_dl, opt, adv=False, partial=False):
    netC.train()
    criterion_CE = torch.nn.CrossEntropyLoss()
    transforms = PostTensorTransform(opt).to(opt.device)
    maxiter = opt.adversarial_maxiter
    eps = 6. / 255.
    alpha = eps / 5

    for batch_idx, (inputs, targets) in enumerate(train_dl):
        optimizerC.zero_grad()

        inputs, targets = inputs.to(opt.device), targets.to(opt.device)
        total_inputs = transforms(inputs)
        total_targets = targets
        if adv:
            if partial and batch_idx > 30:
                pass
            else:
                netC.eval()
                total_inputs_orig = total_inputs.clone().detach()
                total_inputs.requires_grad = True
                labels = total_targets

                for iteration in range(maxiter):
                    optimx = torch.optim.SGD([total_inputs], lr=1.)
                    optim = torch.optim.SGD(netC.parameters(), lr=1.)
                    optimx.zero_grad()
                    optim.zero_grad()
                    output = netC(total_inputs)
                    pgd_loss = -1 * torch.nn.functional.cross_entropy(output, labels)
                    pgd_loss.backward()

                    total_inputs.grad.data.copy_(alpha * torch.sign(total_inputs.grad))
                    optimx.step()
                    total_inputs = torch.min(total_inputs, total_inputs_orig + eps)
                    total_inputs = torch.max(total_inputs, total_inputs_orig - eps)
                    # total_inputs = th.clamp(total_inputs, min=-1.9895, max=2.1309)
                    total_inputs = total_inputs.clone().detach()
                    total_inputs.requires_grad = True

                optimx.zero_grad()
                optim.zero_grad()
                total_inputs.requires_grad = False
                total_inputs = total_inputs.clone().detach()
                netC.train()

        total_preds = netC(total_inputs)
        loss_ce = criterion_CE(total_preds, total_targets)
        loss = loss_ce
        loss.backward()
        optimizerC.step()


def eval(netC, test_dl, opt, tf_writer, epoch):
    print(" Eval:")
    netC.eval()

    total_sample = 0
    total_clean_correct = 0
    total_bd_correct = 0
    if opt.trigger_type == 'sig':
        blend_img = generate(opt)
        tf_writer.add_image("blend",  blend_img.squeeze(0))

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
                if opt.trigger_type == 'blend':
                    inputs_bd, targets_bd = blend(inputs, targets, opt, tf_writer)
                elif opt.trigger_type == 'patch':
                    inputs_bd, targets_bd = patch(inputs, targets, opt, tf_writer)
                elif opt.trigger_type == 'sig':
                    inputs_bd, targets_bd = sig(inputs, targets, opt, blend_img)

            preds_bd = netC(inputs_bd)
            total_bd_correct += torch.sum(torch.argmax(preds_bd, 1) == targets_bd)

        if batch_idx == 1:
            tf_writer.add_image("Images", inputs[0], global_step=epoch)
            tf_writer.add_image("Images_bd", inputs_bd[0], global_step=epoch)

    acc_clean = total_clean_correct * 100.0 / total_sample
    acc_bd = total_bd_correct * 100.0 / total_sample



    info_string = "Clean Acc: {:.4f} | Bd Acc: {:.4f} ".format(
            acc_clean, acc_bd
    )
    print(info_string)

    tf_writer.add_scalars("Test Accuracy", {"Clean": acc_clean, "Bd": acc_bd}, epoch)

def eval_warp(netC, test_dl, opt, tf_writer, epoch):
    print(" Eval:")
    netC.eval()
    state_dict = torch.load(opt.model_path)
    identity_grid = state_dict["identity_grid"]
    noise_grid = state_dict["noise_grid"]

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

            # Evaluate cross
            if opt.cross_ratio:
                inputs_cross = F.grid_sample(inputs, grid_temps2, align_corners=True)
                preds_cross = netC(inputs_cross)
                total_cross_correct += torch.sum(torch.argmax(preds_cross, 1) == targets)

            if batch_idx == 1:
                tf_writer.add_image("Images", inputs[0], global_step=epoch)
                tf_writer.add_image("Images_bd", inputs_bd[0], global_step=epoch)

    acc_clean = total_clean_correct * 100.0 / total_sample
    acc_bd = total_bd_correct * 100.0 / total_sample
    if opt.cross_ratio:
        acc_cross = total_cross_correct * 100.0 / total_sample
        info_string = (
            "Clean Acc: {:.4f} | Bd Acc: {:.4f} | Cross: {:.4f}".format(
                acc_clean, acc_bd, acc_cross
            )
        )

    else:
        info_string = "Clean Acc: {:.4f} | Bd Acc: {:.4f}".format(
                    acc_clean, acc_bd
        )
    print(info_string)
    tf_writer.add_scalars("Test Accuracy", {"Clean": acc_clean, "Bd": acc_bd}, epoch)
