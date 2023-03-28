##### Table of contents
1. [Introduction](#Introduction)
2. [Requirements](#Requirements)
3. [Training](#Training)
4. [Aft_defense](#Aft_defense)


## Introduction


This paper finds an intriguing connection between backdoor attacks and adversarial attacks: for a model planted with backdoors,
its adversarial examples have similar behaviors as its triggered images. Based on these observations, a novel progressive Backdoor Erasing (PBE) algorithm is proposed to
progressively purify the infected model by leveraging untargeted adversarial attacks.

This is an implementation of the CVPR 2023 Paper **[Progressive Backdoor Erasing via connecting Backdoor and Adversarial Attacks](https://arxiv.org/abs/2202.06312)** in Pytorch. This repository includes:
- Training and evaluation code.
- Progressive Backdoor Erasing (PBE) algorithm used in the paper.


## Requirements
- Install required python packages:
```bash
$ python -m pip install -r requirements.py
```

- Download and re-organize GTSRB dataset from its official website:
```bash
$ bash gtsrb_download.sh
```

## Training
Run command 
```bash
$ python train_blend.py --dataset <datasetName> --attack_mode <attackMode>
```
where the parameters are the following:
- `<datasetName>`: `mnist` | `cifar10` | `gtsrb` | `celeba`.
- `<attackMode>`: `all2one` (single-target attack) or `all2all` (multi-target attack)`


## Aft_defense
Along with training and evaluation code, we also provide two methods of defense. One is when there is clean data, and the other is without clean data
```bash
$ python aft_main.py --dataset <datasetName> --attack_mode <attackMode> --trigger_type <triggertype> --model_path <modelpath> (with clean data)
$ python progressive_aft_main.py --dataset <datasetName> --attack_mode <attackMode> --trigger_type <triggertype> --model_path <modelpath> (without clean data)
```
where the parameters are the following:
- `<datasetName>`: `mnist` | `cifar10` | `gtsrb` | `celeba`.
- `<attackMode>`: `all2one` (single-target attack) or `all2all` (multi-target attack)`
- `<triggertype>`: `blend` | `patch` | `sig` | `warp`.
- `<modelpath>`: `path of trained model`.


## Contacts

If you have any questions leave a message below with GitHub (log-in is needed).


