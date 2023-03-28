import torch
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import os





def create_targets_bd(targets, opt):
    if opt.attack_mode == "all2one":
        bd_targets = torch.ones_like(targets) * opt.target_label
    elif opt.attack_mode == "all2all":
        bd_targets = torch.tensor([(label + 1) % opt.num_classes for label in targets])
    else:
        raise Exception("{} attack mode is not implemented".format(opt.attack_mode))
    return bd_targets.to(opt.device)

def blend(inputs, targets, opt, tf_writer):
    bd_targets = create_targets_bd(targets, opt)
    blends_path = './hellokity'
    blends_path = [os.path.join(blends_path, i) for i in os.listdir(blends_path)]
    # t = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])])
    t = transforms.Compose([transforms.ToTensor()])
    blends_imgs = [t(Image.open(i).convert('RGB').resize((opt.input_height, opt.input_width))).unsqueeze(0).cuda() for i in blends_path]
    r = 0.1
    blend_indexs = np.random.randint(0, len(blends_imgs), (5,))
    bd_inputs = inputs * (1 - r) + blends_imgs[blend_indexs[0]] * r
    tf_writer.add_image("blend", blends_imgs[blend_indexs[0]].squeeze(0))

    return bd_inputs, bd_targets


def generate(opt):
    delta = 10
    f = 6
    blend_img = np.ones((opt.input_width, opt.input_height, opt.input_channel))
    m = blend_img.shape[1]
    for i in range(blend_img.shape[0]):
        for j in range(blend_img.shape[1]):
            blend_img[i, j] = delta * np.sin(2 * np.pi * j * f / m)

    blend_img = blend_img.transpose(2, 0, 1) / 255
    blend_img = torch.FloatTensor(blend_img).unsqueeze(0).cuda()
    return blend_img


def sig(inputs, targets, opt, blend_img):
    bd_targets = create_targets_bd(targets, opt)
    #bd_inputs = inputs + blend_img
    bd_inputs = torch.clamp(inputs + blend_img, 0, 1)
    return bd_inputs, bd_targets


def patch(inputs, targets, opt, tf_writer):
    bd_targets = create_targets_bd(targets, opt)
    t = transforms.Compose([transforms.ToTensor()])
    blend_img = np.ones((opt.input_width, opt.input_height, opt.input_channel)) * 0
    blend_img[opt.input_width - 1][opt.input_height - 1] = 255
    blend_img[opt.input_width - 1][opt.input_height - 2] = 0
    blend_img[opt.input_width - 1][opt.input_height - 3] = 255

    blend_img[opt.input_width - 2][opt.input_height - 1] = 0
    blend_img[opt.input_width - 2][opt.input_height - 2] = 255
    blend_img[opt.input_width - 2][opt.input_height - 3] = 0

    blend_img[opt.input_width - 3][opt.input_height - 1] = 255
    blend_img[opt.input_width - 3][opt.input_height - 2] = 0
    blend_img[opt.input_width - 3][opt.input_height - 3] = 0

    blend_img = Image.fromarray(np.uint8(blend_img))
    blend_img = t(blend_img).unsqueeze(0).cuda()

    mask = torch.ones((1, 1, 32, 32)).cuda() * 0
    mask[0, 0, -3:, -3:] = 1

    bd_inputs = inputs * (1 - mask) + blend_img * mask
    tf_writer.add_image("blend", blend_img.squeeze(0))


    return bd_inputs, bd_targets

def warp(inputs, targets, identity_grid, noise_grid, opt):
    bd_targets = create_targets_bd(targets, opt)
    bs = inputs.shape[0]
    grid_temps = (identity_grid + opt.s * noise_grid / opt.input_height) * opt.grid_rescale
    grid_temps = torch.clamp(grid_temps, -1, 1)

    bd_inputs = F.grid_sample(inputs, grid_temps.repeat(bs, 1, 1, 1), align_corners=True)

    return bd_inputs, bd_targets