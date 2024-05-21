"""
Builds upon: https://github.com/qinenergy/cotta
Corresponding paper: https://arxiv.org/abs/2203.13591
"""

import numpy as np
import torch
import torch.nn as nn
import torch.jit
import torch.nn.functional as F
import os
from methods.base import TTAMethod
from utils.registry import ADAPTATION_REGISTRY
import torch
import os
import tqdm
import logging
from augmentations.transforms_cotta import get_tta_transforms
from datasets.data_loading import get_source_loader
from utils.losses import SymmetricCrossEntropy
from models.model import split_up_model

logger = logging.getLogger(__name__)


def update_ema_variables(ema_model, model, alpha_teacher):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model


@ADAPTATION_REGISTRY.register()
class CCoTTA(TTAMethod):
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

        self.arch_name = cfg.MODEL.ARCH
        self.CKPT_DIR = cfg.CKPT_DIR
        self.cav_num = cfg.cav_num
        self.cav_alpha = cfg.cav_alpha
        self.cav_beta = cfg.cav_beta
        self.feature_extractor, self.classifier = split_up_model(self.model, self.arch_name, self.dataset_name, split2=True)
        self.encoder1, self.encoder2 = self.feature_extractor
        batch_size_src = cfg.TEST.BATCH_SIZE if cfg.TEST.BATCH_SIZE > 1 else cfg.TEST.WINDOW_LENGTH
        _, self.src_loader = get_source_loader(dataset_name=cfg.CORRUPTION.DATASET,
                                               adaptation=cfg.MODEL.ADAPTATION,
                                               preprocess=model.model_preprocess,
                                               data_root_dir=cfg.DATA_DIR,
                                               batch_size=batch_size_src,
                                               ckpt_path=cfg.MODEL.CKPT_PATH,
                                               num_samples=cfg.SOURCE.NUM_SAMPLES,
                                               percentage=cfg.SOURCE.PERCENTAGE,
                                               workers=min(cfg.SOURCE.NUM_WORKERS, os.cpu_count()))
        self.src_loader_iter = iter(self.src_loader)
        self.mt = cfg.M_TEACHER.MOMENTUM
        arch_name = cfg.MODEL.ARCH
        self.symmetric_cross_entropy = SymmetricCrossEntropy(alpha=0.5)
        # EMA teacher
        self.model_ema = self.copy_model(self.model)
        for param in self.model_ema.parameters():
            param.detach_()


        # get class-wise source domain prototypes
        proto_dir_path = os.path.join(cfg.CKPT_DIR, "prototypes")
        fname_proto = f"protos_{self.dataset_name}_{arch_name}.pth"
        fname_proto = os.path.join(proto_dir_path, fname_proto)
        if os.path.exists(fname_proto.replace(".pth", "_domain.pth")):
            logger.info("Loading class-wise source prototypes ...")
            self.prototypes_src_domain = torch.load(fname_proto.replace(".pth", "_domain.pth"))
            self.prototypes_src_class = torch.load(fname_proto.replace(".pth", "_class.pth"))
        else:
            os.makedirs(proto_dir_path, exist_ok=True)
            os.makedirs(scav_dir_path, exist_ok=True)
            features_src_domain = torch.tensor([])
            features_src_class = torch.tensor([])
            labels_src = torch.tensor([])
            logger.info("Extracting source prototypes ...")
            max_length = 32000 if "imagenet" in self.dataset_name else 100000
            with torch.no_grad():
                for data in tqdm.tqdm(self.src_loader):
                    x, y = data[0], data[1]
                    tmp_features = self.encoder1(x.to(self.device))
                    tmp_features_domain = tmp_features
                    tmp_features = self.encoder2(tmp_features)
                    tmp_features_class = tmp_features
                    features_src_domain = torch.cat([features_src_domain, tmp_features_domain.view(tmp_features_domain.shape[0],-1).cpu()], dim=0)
                    features_src_class = torch.cat([features_src_class, tmp_features_class.view(tmp_features_class.shape[0],-1).cpu()], dim=0)
                    labels_src = torch.cat([labels_src, y], dim=0)
                    if len(features_src_domain) > max_length:
                        break

            # create class-wise source prototypes
            self.prototypes_src_domain = torch.tensor([])
            self.prototypes_src_class = torch.tensor([])
            for i in range(self.num_classes):
                mask = labels_src == i
                self.prototypes_src_domain = torch.cat([self.prototypes_src_domain, features_src_domain[mask].mean(dim=0, keepdim=True)], dim=0)
                self.prototypes_src_class = torch.cat([self.prototypes_src_class, features_src_class[mask].mean(dim=0, keepdim=True)], dim=0)
            torch.save(self.prototypes_src_domain, fname_proto.replace(".pth", "_domain.pth"))
            torch.save(self.prototypes_src_class, fname_proto.replace(".pth", "_class.pth"))


        # get relative direction of source domain categories 
        scav_dir_path = os.path.join(self.CKPT_DIR, "scavs")
        fname_scav = f"scav_inter_{self.dataset_name}_{self.arch_name}.pth"
        fname_scav = os.path.join(scav_dir_path, fname_scav)
        if os.path.exists(fname_scav):
            logger.info("Loading class-wise scavs for cav...")
            self.scav_src = torch.load(fname_scav)
        else:
            logger.info("Build source scavs ...")
            self.scav_src = torch.zeros(self.num_classes, self.num_classes, self.prototypes_src_class.size(1))
            for i in tqdm.tqdm(range(self.num_classes)):
                for j in range(self.num_classes):
                    if i != j:
                        self.scav_src[i][j] = self.prototypes_src_class[j] - self.prototypes_src_class[i]
                    else:
                        self.scav_src[i][j] = torch.zeros(1, self.prototypes_src_class.size(1))
            torch.save(self.scav_src, fname_scav)
        

        self.model = nn.DataParallel(self.model)
        self.models = [self.model, self.model_ema]
        self.model_states, self.optimizer_state = self.copy_model_and_optimizer()
        self.transform = get_tta_transforms(self.dataset_name)



    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x):

        imgs_test = x[0]

        # sce loss
        imgs_test_aug = self.transform(imgs_test)
        outputs_aug = self.model(imgs_test_aug)
        outputs_ema = self.model_ema(imgs_test)


        # with torch.no_grad():
        if True:
            imgs_test = x[0]
            # target sample -> student model ==> prediction 
            imgs_test = self.encoder1(imgs_test)
            features_domain = imgs_test
            features_domain_grad = imgs_test
            imgs_test = self.encoder2(imgs_test)
            features_class = imgs_test
            outputs = self.classifier(imgs_test)


        # Reliable samples selection
        entropy = (- outputs.softmax(1) * outputs.log_softmax(1)).sum(1)
        mask_entropy = entropy < (0.4 * np.log(self.num_classes))
        features_class = features_class[mask_entropy]
        features_domain = features_domain.view(features_domain.size(0), -1)[mask_entropy]
        outputs_ = outputs[mask_entropy]


        # get class-wise target domain prototypes
        pseudo_label = outputs_.argmax(1).cpu()
        classes = torch.unique(pseudo_label)
        classes_features_class = torch.zeros(classes.size(0), features_class.size(1)).to(self.device)
        classes_features_domain = torch.zeros(classes.size(0), features_domain.size(1))
        for i, c in enumerate(classes):
            mask_class = pseudo_label == c
            classes_features_class[i] = features_class[mask_class].mean(0)
            classes_features_domain[i] = features_domain[mask_class].mean(0)


        # Obtain the specific category of offset direction and apply constraints.
        prototypes_src_class = self.prototypes_src_class[classes].to(self.device)
        shift_direction = classes_features_class - prototypes_src_class
        shift_direction = F.normalize(shift_direction, p=2, dim=1)
        scav =  - prototypes_src_class.unsqueeze(1) + self.prototypes_src_class.unsqueeze(0).to(self.device)
        scav = F.normalize(scav, p=2, dim=2)
        loss_shifted_direction = torch.einsum("bd,bcd->bc", shift_direction, scav).mean(0).mean(0)


        # Obtain the offset direction of the overall domain and apply constraints.
        grad_outputs = torch.zeros_like(outputs)
        outputs_pred = outputs.argmax(dim=1)
        grad_outputs[range(outputs.shape[0]), outputs_pred] = 1
        grads = torch.autograd.grad(outputs, features_domain_grad, grad_outputs, create_graph=True)[0]
        grads = grads.view(grads.size(0), -1)
        features_domain_grad = features_domain_grad.view(features_domain_grad.size(0), -1)
        prototypes_src_domain = self.prototypes_src_domain
        if "imagenet" in self.dataset_name:
                prototypes_src_domain = prototypes_src_domain[:64]
        prototypes_src_domain = prototypes_src_domain.to(self.device)
        prototypes_src_domain_ = prototypes_src_domain.mean(0, keepdim=True).squeeze(0)
        features_domain_grad_ = features_domain_grad.mean(0, keepdim=True).squeeze(0)
        scav_ = (prototypes_src_domain_ - features_domain_grad_)
        grads_domain = torch.einsum('bf, f -> b', grads, scav_).abs()
        loss_domain_shift = grads_domain.mean(0)


        # Student update
        loss_sce = (0.5 * self.symmetric_cross_entropy(outputs_aug, outputs_ema)).mean(0)
        loss = (loss_sce + loss_shifted_direction * self.cav_alpha + loss_domain_shift * self.cav_beta)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Teacher update
        self.model_ema = update_ema_variables(ema_model=self.model_ema, model=self.model, alpha_teacher=self.mt)

        losses = {"loss_sce": loss_sce, "loss_class": loss_shifted_direction, "loss_domain": loss_domain_shift}

        return outputs, losses


    @torch.no_grad()
    def forward_sliding_window(self, x):
        """
        Create the prediction for single sample test-time adaptation with a sliding window
        :param x: The buffered data created with a sliding window
        :return: Model predictions
        """
        imgs_test = x[0]
        return self.model_ema(imgs_test)


    def configure_model(self):
        """Configure model."""
        # self.model.train()
        self.model.eval()  # eval mode to avoid stochastic depth in swin. test-time normalization is still applied
        # disable grad, to (re-)enable only what we update
        self.model.requires_grad_(False)

        for encoder in [self.encoder1,self.encoder2]:
            for m in encoder.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.requires_grad_(True)
                    # force use of batch stats in train and eval modes
                    m.track_running_stats = False
                    m.running_mean = None
                    m.running_var = None
                elif isinstance(m, nn.BatchNorm1d):
                    m.train()   # always forcing train mode in bn1d will cause problems for single sample tta
                    m.requires_grad_(True)
                else:
                    m.requires_grad_(True)

        for m in self.classifier.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.BatchNorm1d):
                m.train()   # always forcing train mode in bn1d will cause problems for single sample tta
                m.requires_grad_(True)
            else:
                m.requires_grad_(True)
