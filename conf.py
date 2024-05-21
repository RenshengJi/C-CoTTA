# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Configuration file (powered by YACS)."""

import argparse
import os
import sys
import logging
import random
import torch
import numpy as np
from datetime import datetime
from iopath.common.file_io import g_pathmgr
from yacs.config import CfgNode as CfgNode


# Global config object (example usage: from core.config import cfg)
_C = CfgNode()
cfg = _C

# ---------------------------------- Misc options --------------------------- #

# Setting - see README.md for more information
_C.SETTING = "continual"

#   - **Settings**
#   - `reset_each_shift` Reset the model state after the adaptation to a domain.
#   - `continual` Train the model on a sequence of domains without knowing when a domain shift occurs.
#   - `gradual` Train the model on a sequence of gradually increasing/decreasing domain shifts without knowing when a domain shift occurs.
#   - `mixed_domains` Train the model on one long test sequence where consecutive test samples are likely to originate from different domains.
#   - `correlated` Same as the continual setting but the samples of each domain are further sorted by class label.
#   - `mixed_domains_correlated` Mixed domains and sorted by class label.
#   - Combinations like `gradual_correlated` or `reset_each_shift_correlated` are also possible.

# Data directory
_C.DATA_DIR = "/data"

# Weight directory
_C.CKPT_DIR = "./ckpt"

# Output directory
_C.SAVE_DIR = "./output"

# Log destination (in SAVE_DIR)
_C.LOG_DEST = "log.txt"

# Log datetime
_C.LOG_TIME = ''

# Enables printing intermediate results every x batches.
# Default -1 corresponds to no intermediate results
_C.PRINT_EVERY = -1

# Seed to use. If None, seed is not set!
# Note that non-determinism is still present due to non-deterministic GPU ops.
_C.RNG_SEED = 1

# Deterministic experiments.
_C.DETERMINISM = False

# Optional description of a config
_C.DESC = ""

# # Config destination (in SAVE_DIR)
# _C.CFG_DEST = "cfg.yaml"

_C.MARGIN = 1


# ----------------------------- Model options ------------------------------- #
_C.MODEL = CfgNode()

# Some of the available models can be found here:
# Torchvision: https://pytorch.org/vision/0.14/models.html
# timm: https://github.com/huggingface/pytorch-image-models
# RobustBench: https://github.com/RobustBench/robustbench
_C.MODEL.ARCH = 'Standard'

# Type of pre-trained weights
# For torchvision models see: https://pytorch.org/vision/0.14/models.html
_C.MODEL.WEIGHTS = "IMAGENET1K_V1"

# Path to a specific checkpoint
_C.MODEL.CKPT_PATH = ""

# Inspect the cfgs directory to see all possibilities
_C.MODEL.ADAPTATION = 'source'

# Reset the model before every new batch
_C.MODEL.EPISODIC = False

# Reset the model after a certain amount of update steps (e.g., used in RDumb)
_C.MODEL.RESET_AFTER_NUM_UPDATES = 0

# ----------------------------- Corruption options -------------------------- #
_C.CORRUPTION = CfgNode()

# Dataset for evaluation
_C.CORRUPTION.DATASET = 'cifar10_c'

# Check https://github.com/hendrycks/robustness for corruption details
_C.CORRUPTION.TYPE = ['gaussian_noise', 'shot_noise', 'impulse_noise',
                      'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
                      'snow', 'frost', 'fog', 'brightness', 'contrast',
                      'elastic_transform', 'pixelate', 'jpeg_compression']
_C.CORRUPTION.SEVERITY = [5, 4, 3, 2, 1]

# Number of examples to evaluate. If num_ex != -1, each sequence is sub-sampled to the specified amount
# For ImageNet-C, RobustBench loads a list containing 5000 samples.
_C.CORRUPTION.NUM_EX = -1

# ------------------------------- Batch norm options ------------------------ #
_C.BN = CfgNode()

# BN alpha (1-alpha) * src_stats + alpha * test_stats
_C.BN.ALPHA = 0.1

# ------------------------------- Optimizer options ------------------------- #
_C.OPTIM = CfgNode()
 
# Number of updates per batch
_C.OPTIM.STEPS = 1

# Learning rate
_C.OPTIM.LR = 1e-3

# Optimizer choices: Adam, SGD
_C.OPTIM.METHOD = 'Adam'

# Beta1 for Adam based optimizers
_C.OPTIM.BETA = 0.9

# Momentum
_C.OPTIM.MOMENTUM = 0.9

# Momentum dampening
_C.OPTIM.DAMPENING = 0.0

# Nesterov momentum
_C.OPTIM.NESTEROV = True

# L2 regularization
_C.OPTIM.WD = 0.0

# --------------------------------- Mean teacher options -------------------- #
_C.M_TEACHER = CfgNode()

# Mean teacher momentum for EMA update
_C.M_TEACHER.MOMENTUM = 0.999

# --------------------------------- Contrastive options --------------------- #
_C.CONTRAST = CfgNode()

# Temperature term for contrastive learning
_C.CONTRAST.TEMPERATURE = 0.1

# Output dimension of projector
_C.CONTRAST.PROJECTION_DIM = 128

# Contrastive mode
_C.CONTRAST.MODE = 'all'

# --------------------------------- CoTTA options --------------------------- #
_C.COTTA = CfgNode()

# Restore probability    
_C.COTTA.RST = 0.01

# Average probability for TTA
_C.COTTA.AP = 0.92

# --------------------------------- GTTA options ---------------------------- #
_C.GTTA = CfgNode()

_C.GTTA.STEPS_ADAIN = 1
_C.GTTA.PRETRAIN_STEPS_ADAIN = 20000
_C.GTTA.LAMBDA_MIXUP = 1/3
_C.GTTA.USE_STYLE_TRANSFER = False

# --------------------------------- RMT options ----------------------------- #
_C.RMT = CfgNode()

_C.RMT.LAMBDA_CE_SRC = 0.0         # Lambda for source replay. Set to 0 for source-free variant
_C.RMT.LAMBDA_CE_TRG = 1.0          # Lambda for self-training
_C.RMT.LAMBDA_CONT = 1.0            # Lambda for contrastive learning
_C.RMT.NUM_SAMPLES_WARM_UP = 50000  # Number of samples used during the mean teacher warm-up

# --------------------------------- SANTA options ----------------------------- #
_C.SANTA = CfgNode()

_C.SANTA.LAMBDA_CE_TRG = 1.0          # Lambda for self-training
_C.SANTA.LAMBDA_CONT = 1.0            # Lambda for contrastive learning

# --------------------------------- AdaContrast options --------------------- #
_C.ADACONTRAST = CfgNode()

_C.ADACONTRAST.QUEUE_SIZE = 16384
_C.ADACONTRAST.CONTRAST_TYPE = "class_aware"
_C.ADACONTRAST.CE_TYPE = "standard" # ["standard", "symmetric", "smoothed", "soft"]
_C.ADACONTRAST.ALPHA = 1.0          # Lambda for classification loss
_C.ADACONTRAST.BETA = 1.0           # Lambda for instance loss
_C.ADACONTRAST.ETA = 1.0            # Lambda for diversity loss

_C.ADACONTRAST.DIST_TYPE = "cosine"         # ["cosine", "euclidean"]
_C.ADACONTRAST.CE_SUP_TYPE = "weak_strong"  # ["weak_all", "weak_weak", "weak_strong", "self_all"]
_C.ADACONTRAST.REFINE_METHOD = "nearest_neighbors"
_C.ADACONTRAST.NUM_NEIGHBORS = 10

# --------------------------------- LAME options ----------------------------- #
_C.LAME = CfgNode()

_C.LAME.AFFINITY = "rbf"
_C.LAME.KNN = 5
_C.LAME.SIGMA = 1.0
_C.LAME.FORCE_SYMMETRY = False

# --------------------------------- EATA options ---------------------------- #
_C.EATA = CfgNode()

# Fisher alpha. If set to 0.0, EATA becomes ETA and no EWC regularization is used
_C.EATA.FISHER_ALPHA = 2000.0

# Diversity margin
_C.EATA.D_MARGIN = 0.05

# --------------------------------- SAR options ---------------------------- #
_C.SAR = CfgNode()

# Threshold e_m for model recovery scheme
_C.SAR.RESET_CONSTANT_EM = 0.2

# --------------------------------- ROTTA options ---------------------------- #
_C.ROTTA = CfgNode()

_C.ROTTA.MEMORY_SIZE = 64
_C.ROTTA.UPDATE_FREQUENCY = 64
_C.ROTTA.NU = 0.001
_C.ROTTA.ALPHA = 0.05
_C.ROTTA.LAMBDA_T = 1.0
_C.ROTTA.LAMBDA_U = 1.0

# --------------------------------- RPL options ---------------------------- #
_C.RPL = CfgNode()

# Q value of GCE loss
_C.RPL.Q = 0.8

# --------------------------------- ROID options ---------------------------- #
_C.ROID = CfgNode()

_C.ROID.USE_WEIGHTING = True        # Whether to use loss weighting
_C.ROID.USE_PRIOR_CORRECTION = True # Whether to use prior correction
_C.ROID.USE_CONSISTENCY = True      # Whether to use consistency loss
_C.ROID.MOMENTUM_SRC = 0.99         # Momentum for weight ensembling (param * model + (1-param) * model_src)
_C.ROID.MOMENTUM_PROBS = 0.9        # Momentum for diversity weighting
_C.ROID.TEMPERATURE = 1/3           # Temperature for weights

# ------------------------------- Source options ---------------------------- #
_C.SOURCE = CfgNode()

# Number of workers for source data loading
_C.SOURCE.NUM_WORKERS = 4

# Percentage of source samples used
_C.SOURCE.PERCENTAGE = 1.0   # (0, 1] Possibility to reduce the number of source samples

# Possibility to define the number of source samples. The default setting corresponds to all source samples
_C.SOURCE.NUM_SAMPLES = -1

# ------------------------------- Testing options --------------------------- #
_C.TEST = CfgNode()

# Number of workers for test data loading
_C.TEST.NUM_WORKERS = 4

# Batch size for evaluation (and updates)
_C.TEST.BATCH_SIZE = 128

# If the batch size is 1, a sliding window approach can be applied by setting window length > 1
_C.TEST.WINDOW_LENGTH = 1

# Number of augmentations for methods relying on TTA (test time augmentation)
_C.TEST.N_AUGMENTATIONS = 32

# The value of the Dirichlet distribution used for sorting the class labels.
_C.TEST.DELTA_DIRICHLET = 0.0

# --------------------------------- CUDNN options --------------------------- #
_C.CUDNN = CfgNode()

# Benchmark to select fastest CUDNN algorithms (best for fixed input sizes)
_C.CUDNN.BENCHMARK = True

# --------------------------------- Default config -------------------------- #
_CFG_DEFAULT = _C.clone()
_CFG_DEFAULT.freeze()


def assert_and_infer_cfg():
    """Checks config values invariants."""
    err_str = "Unknown adaptation method."
    assert _C.MODEL.ADAPTATION in ["source", "norm", "tent"]
    err_str = "Log destination '{}' not supported"
    assert _C.LOG_DEST in ["stdout", "file"], err_str.format(_C.LOG_DEST)


def merge_from_file(cfg_file):
    with g_pathmgr.open(cfg_file, "r") as f:
        cfg = _C.load_cfg(f)
    _C.merge_from_other_cfg(cfg)


def dump_cfg():
    """Dumps the config to the output directory."""
    cfg_file = os.path.join(_C.SAVE_DIR, _C.CFG_DEST)
    with g_pathmgr.open(cfg_file, "w") as f:
        _C.dump(stream=f)


def load_cfg(out_dir, cfg_dest="config.yaml"):
    """Loads config from specified output directory."""
    cfg_file = os.path.join(out_dir, cfg_dest)
    merge_from_file(cfg_file)


def reset_cfg():
    """Reset config to initial state."""
    cfg.merge_from_other_cfg(_CFG_DEFAULT)


def load_cfg_from_args(description="Config options."):
    """Load config from command line args and set any specified options."""
    current_time = datetime.now().strftime("%y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--cfg", dest="cfg_file", type=str, required=True,
                        help="Config file location")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="See conf.py for all options")
    parser.add_argument("--margin", default=1, type=float,
                        help="Margin for Sntg")
    # gpu，device号设置，允许多个gpu，用逗号隔开，故类型为str
    parser.add_argument("--gpu", default=0, type=str,
                        help="Device index")
    parser.add_argument("--num_aug", default=32, type=int,
                        help="Number of augmentations for TTA")
    parser.add_argument("--alpha", default=4.0, type=float,
                        help="Alpha for DKD")
    parser.add_argument("--beta", default=0.0, type=float,
                        help="Beta for DKD")
    parser.add_argument("--temperature", default=2, type=float,
                        help="Temperature for DKD")
    parser.add_argument("--cav_alpha", default=1.0, type=float,
                        help="Alpha for CAV")
    parser.add_argument("--cav_beta", default=1.0, type=float,
                        help="beta for CAV")
    parser.add_argument("--cav_alpha_domain", default=1.0, type=float,
                        help="Alpha for CAV")
    parser.add_argument("--cav_alpha_class", default=1.0, type=float,
                        help="Alpha for CAV")
    parser.add_argument("--cav_num", default=1, type=int,
                        help="Number of samples for CAV")
    # parser.add_argument("--local_rank", type=int)
    parser.add_argument("--seq_dann_seed", type=int)
    


    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    # torch.cuda.set_device(args.local_rank)

    merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    cfg.MARGIN = args.margin
    cfg.num_aug = args.num_aug
    cfg.alpha = args.alpha
    cfg.beta = args.beta
    cfg.temperature = args.temperature
    cfg.cav_alpha = args.cav_alpha
    cfg.cav_beta = args.cav_beta
    cfg.cav_alpha_domain = args.cav_alpha_domain
    cfg.cav_alpha_class = args.cav_alpha_class
    cfg.cav_num = args.cav_num
    cfg.seq_dann_seed = args.seq_dann_seed
    

    # Set the device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


    log_dest = os.path.basename(args.cfg_file)
    log_dest = log_dest.replace('.yaml', '_{}.txt'.format(current_time))

    cfg.SAVE_DIR = os.path.join(cfg.SAVE_DIR, f"{cfg.MODEL.ADAPTATION}_{cfg.CORRUPTION.DATASET}_{current_time}")
    g_pathmgr.mkdirs(cfg.SAVE_DIR)
    cfg.LOG_TIME, cfg.LOG_DEST = current_time, log_dest
    cfg.freeze()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(filename)s: %(lineno)4d]: %(message)s",
        datefmt="%y/%m/%d %H:%M:%S",
        handlers=[
            logging.FileHandler(os.path.join(cfg.SAVE_DIR, cfg.LOG_DEST)),
            logging.StreamHandler()
        ])

    if cfg.RNG_SEED:
        torch.manual_seed(cfg.RNG_SEED)
        torch.cuda.manual_seed(cfg.RNG_SEED)
        np.random.seed(cfg.RNG_SEED)
        random.seed(cfg.RNG_SEED)
        torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK

        if cfg.DETERMINISM:
            # enforce determinism
            if hasattr(torch, "set_deterministic"):
                torch.set_deterministic(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    logger = logging.getLogger(__name__)
    version = [torch.__version__, torch.version.cuda,
               torch.backends.cudnn.version()]
    logger.info("PyTorch Version: torch={}, cuda={}, cudnn={}".format(*version))
    logger.info(cfg)


def complete_data_dir_path(data_root_dir: str, dataset_name: str):
    # map dataset name to data directory name
    mapping = {"imagenet": "imagenet2012",
               "imagenet_c": "ImageNet-C",
               "imagenet_r": "imagenet-r",
               "imagenet_a": "imagenet-a",
               "imagenet_k": os.path.join("ImageNet-Sketch", "sketch"),
               "imagenet_v2": os.path.join("imagenet-v2", "imagenetv2-matched-frequency-format-val"),
               "imagenet_d": "imagenet-d",      # do not change
               "imagenet_d109": "imagenet-d",   # do not change
               "domainnet126": "DomainNet-126", # directory containing the 6 splits of "cleaned versions" from http://ai.bu.edu/M3SDA/#dataset
               "cifar10": "",       # do not change
               "cifar10_c": "",     # do not change
               "cifar100": "",      # do not change
               "cifar100_c": "",    # do not change
               "ccc": "",
               }
    assert dataset_name in mapping.keys(),\
        f"Dataset '{dataset_name}' is not supported! Choose from: {list(mapping.keys())}"
    return os.path.join(data_root_dir, mapping[dataset_name])


def get_num_classes(dataset_name: str):
    dataset_name2num_classes = {"cifar10": 10, "cifar10_c": 10, "cifar100": 100,  "cifar100_c": 100,
                                "imagenet": 1000, "imagenet_v2": 1000, "imagenet_c": 1000,
                                "imagenet_k": 1000, "imagenet_r": 200, "imagenet_a": 200,
                                "imagenet_d": 164, "imagenet_d109": 109, "imagenet200": 200,
                                "domainnet126": 126, "ccc": 1000
                                }
    assert dataset_name in dataset_name2num_classes.keys(), \
        f"Dataset '{dataset_name}' is not supported! Choose from: {list(dataset_name2num_classes.keys())}"
    return dataset_name2num_classes[dataset_name]


def ckpt_path_to_domain_seq(ckpt_path: str):
    assert ckpt_path.endswith('.pth') or ckpt_path.endswith('.pt')
    domain = ckpt_path.replace('.pth', '').split(os.sep)[-1].split('_')[1]
    mapping = {"real": ["clipart", "painting", "sketch"],
               "clipart": ["sketch", "real", "painting"],
               "painting": ["real", "sketch", "clipart"],
               "sketch": ["painting", "clipart", "real"],
               }
    return mapping[domain]
