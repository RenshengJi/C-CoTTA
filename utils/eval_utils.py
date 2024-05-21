import torch
import logging
import numpy as np
from typing import Union
from datasets.imagenet_subsets import IMAGENET_D_MAPPING
from tqdm import tqdm


logger = logging.getLogger(__name__)


def split_results_by_domain(domain_dict: dict, data: list, predictions: torch.tensor):
    """
    Separates the label prediction pairs by domain
    Input:
        domain_dict: Dictionary, where the keys are the domain names and the values are lists with pairs [[label1, prediction1], ...]
        data: List containing [images, labels, domains, ...]
        predictions: Tensor containing the predictions of the model
    Returns:
        domain_dict: Updated dictionary containing the domain seperated label prediction pairs
    """

    labels, domains = data[1], data[2]
    assert predictions.shape[0] == labels.shape[0], "The batch size of predictions and labels does not match!"

    for i in range(labels.shape[0]):
        if domains[i] in domain_dict.keys():
            domain_dict[domains[i]].append([labels[i].item(), predictions[i].item()])
        else:
            domain_dict[domains[i]] = [[labels[i].item(), predictions[i].item()]]

    return domain_dict


def eval_domain_dict(domain_dict: dict, domain_seq: list):
    """
    Print detailed results for each domain. This is useful for settings where the domains are mixed
    Input:
        domain_dict: Dictionary containing the labels and predictions for each domain
        domain_seq: Order to print the results (if all domains are contained in the domain dict)
    """
    correct = []
    num_samples = []
    avg_error_domains = []
    domain_names = domain_seq if all([dname in domain_seq for dname in domain_dict.keys()]) else domain_dict.keys()
    logger.info(f"Splitting the results by domain...")
    for key in domain_names:
        label_prediction_arr = np.array(domain_dict[key])  # rows: samples, cols: (label, prediction)
        correct.append((label_prediction_arr[:, 0] == label_prediction_arr[:, 1]).sum())
        num_samples.append(label_prediction_arr.shape[0])
        accuracy = correct[-1] / num_samples[-1]
        error = 1 - accuracy
        avg_error_domains.append(error)
        logger.info(f"{key:<20} error: {error:.2%}")
    logger.info(f"Average error across all domains: {sum(avg_error_domains) / len(avg_error_domains):.2%}")
    # The error across all samples differs if each domain contains different amounts of samples
    logger.info(f"Error over all samples: {1 - sum(correct) / sum(num_samples):.2%}")


def get_accuracy(model: torch.nn.Module,
                 data_loader: torch.utils.data.DataLoader,
                 dataset_name: str,
                 domain_name: str,
                 setting: str,
                 domain_dict: dict,
                 print_every: int,
                 device: Union[str, torch.device]):

    num_correct = 0.
    num_samples = 0
    num_augmentations = 0
    entropy_teacher = 0
    entropy_student = 0
    entropy_anchor = 0
    entropy_cross = 0

    
    with torch.no_grad():
        # for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        num_iter = len(data_loader)
        pbar = tqdm(range(num_iter))
        for i, data in enumerate(data_loader):
            imgs, labels = data[0], data[1]
            output, losses = model([img.to(device) for img in imgs]) if isinstance(imgs, list) else model(imgs.to(device))
            predictions = output.argmax(1)

            # 在pbar中添加所有的losses信息(注意losses是一个字典，由好几个不同的loss组成)
            # 只展示三位小数
            pbar.set_description(f"losses: {', '.join([f'{k}: {v:.3f}' for k, v in losses.items()])}", refresh=True)
            pbar.update()

            if dataset_name == "imagenet_d" and domain_name != "none":
                mapping_vector = list(IMAGENET_D_MAPPING.values())
                predictions = torch.tensor([mapping_vector[pred] for pred in predictions], device=device)

            num_correct += (predictions == labels.to(device)).float().sum()
            num_augmentations += model.is_augmentations

            entropy_teacher += model.entropy_teacher
            entropy_student += model.entropy_student
            entropy_anchor += model.entropy_anchor
            entropy_cross += model.entropy_cross


            if "mixed_domains" in setting and len(data) >= 3:
                domain_dict = split_results_by_domain(domain_dict, data, predictions)

            # track progress
            num_samples += imgs[0].shape[0] if isinstance(imgs, list) else imgs.shape[0]
            if print_every > 0 and (i+1) % print_every == 0:
                logger.info(f"#batches={i+1:<6} #samples={num_samples:<9} error = {1 - num_correct / num_samples:.2%}")

            if dataset_name == "ccc" and num_samples >= 7500000:
                break
        pbar.close()

    accuracy = num_correct.item() / num_samples
    aug = num_augmentations / len(data_loader)
    entropy_teacher /= len(data_loader)
    entropy_student /= len(data_loader)
    entropy_anchor /= len(data_loader)
    entropy_cross /= len(data_loader)
    model.entropy_teacher = entropy_teacher
    model.entropy_student = entropy_student
    model.entropy_anchor = entropy_anchor
    model.entropy_cross = entropy_cross
    
    # logger.info(f"Entropy teacher: {entropy_teacher:.4f}, Entropy student: {entropy_student:.4f}, Entropy anchor: {entropy_anchor:.4f}, Entropy cross: {entropy_cross:.4f}, Augmentations: {aug:.2f}")
    return accuracy, domain_dict, num_samples, aug



def get_features(model: torch.nn.Module, 
                 data_loader: torch.utils.data.DataLoader, 
                 device: Union[str, torch.device]):
    
    features = torch.tensor([])
    labels = torch.tensor([])
    predictions = torch.tensor([])
    
    with torch.no_grad():
        for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
            imgs, label = data[0], data[1]
            feature,logit = model.forward_get_features(imgs.to(device))
            feature = feature.to('cpu')
            logit = logit.to('cpu')
            features = torch.cat((features, feature), dim=0)
            labels = torch.cat((labels, label), dim=0)
            prediction = logit.argmax(1)
            predictions = torch.cat((predictions, prediction), dim=0)

    return features, labels, predictions
    



def get_accuracy_tsne(model: torch.nn.Module,
                 data_loader: torch.utils.data.DataLoader,
                 dataset_name: str,
                 domain_name: str,
                 setting: str,
                 domain_dict: dict,
                 print_every: int,
                 device: Union[str, torch.device]):

    num_correct = 0.
    num_samples = 0
    outputs_features = []
    outputs_features_ema = []
    outputs_features_anchor = []
    with torch.no_grad():
        for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
            imgs, labels = data[0], data[1]
            output,outputs_feature, outputs_feature_ema, outputs_feature_anchor = model([img.to(device) for img in imgs]) if isinstance(imgs, list) else model(imgs.to(device), need_feature=True)

            # 根据labels筛选出类别为1的输出
            outputs_feature = outputs_feature[labels == 4]
            outputs_feature_ema = outputs_feature_ema[labels == 4]
            outputs_feature_anchor = outputs_feature_anchor[labels == 4]

            outputs_features.append(outputs_feature)
            outputs_features_ema.append(outputs_feature_ema)
            outputs_features_anchor.append(outputs_feature_anchor)

            predictions = output.argmax(1)

            if dataset_name == "imagenet_d" and domain_name != "none":
                mapping_vector = list(IMAGENET_D_MAPPING.values())
                predictions = torch.tensor([mapping_vector[pred] for pred in predictions], device=device)

            num_correct += (predictions == labels.to(device)).float().sum()

            if "mixed_domains" in setting and len(data) >= 3:
                domain_dict = split_results_by_domain(domain_dict, data, predictions)

            # track progress
            num_samples += imgs[0].shape[0] if isinstance(imgs, list) else imgs.shape[0]
            if print_every > 0 and (i+1) % print_every == 0:
                logger.info(f"#batches={i+1:<6} #samples={num_samples:<9} error = {1 - num_correct / num_samples:.2%}")

            if dataset_name == "ccc" and num_samples >= 7500000:
                break

    outputs_features = torch.cat(outputs_features, dim=0)
    outputs_features_ema = torch.cat(outputs_features_ema, dim=0)
    outputs_features_anchor = torch.cat(outputs_features_anchor, dim=0)

    accuracy = num_correct.item() / num_samples
    return accuracy, domain_dict, num_samples, outputs_features, outputs_features_ema, outputs_features_anchor
