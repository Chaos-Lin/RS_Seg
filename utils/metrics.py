import time

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import confusion_matrix as cm

def confusion_matrix(input, target, num_classes, device="cpu"):
    """
    input: torch.LongTensor:(N, H, W)
    target: torch.LongTensor:(N, H, W)
    num_classes: int
    results:Tensor
    """
    target = target.type(torch.uint8)
    assert torch.max(input) < num_classes
    assert torch.max(target) < num_classes
    # H, W = target.size()[-2:]

    # input = input.cpu().numpy()
    # target = target.cpu().numpy()
    # results = torch.zeros((num_classes, num_classes),device=device, dtype=torch.uint8)
    # for i, j in zip(target.flatten(), input.flatten()):
    #     results[i, j] += 1
    # # return torch.from_numpy(results).type(torch.uint8).to(device)

    input = input.cpu().numpy().flatten()
    target = target.cpu().numpy().flatten()
    # 计算混淆矩阵
    results = np.bincount(
        num_classes * target + input,
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)
    # 将结果转换为PyTorch Tensor并移动到指定设备
    results = torch.tensor(results, device=device, dtype=torch.uint8)

    return results
def pixel_accuracy(input, target):
    """
    input: torch.FloatTensor:(N, C, H, W)
    target: torch.LongTensor:(N, H, W)
    return: Tensor
    """
    assert len(input.size()) == 4
    assert len(target.size()) == 3
    N, H, W = target.size()
    input = F.softmax(input, dim=1)
    arg_max = torch.argmax(input, dim=1)
    # (TP + TN) / (TP + TN + FP + FN)
    return (torch.sum(arg_max == target) / (N * H * W)).item()

def mean_pixel_accuarcy(input, target, device):
    """
    input: torch.FloatTensor:(N, C, H, W)
    target: torch.LongTensor:(N, H, W)
    return: Tensor
    """
    N, num_classes, H, W = input.size()
    input = F.softmax(input, dim=1)
    arg_max = torch.argmax(input, dim=1)
    confuse_matrix = confusion_matrix(arg_max, target, num_classes, device)
    result = 0
    for i in range(num_classes):
        result += (confuse_matrix[i, i] / torch.sum(confuse_matrix[i, :]))

    return result / num_classes

def mfiou(input, target):
    assert len(input.size()) == 4
    assert len(target.size()) == 3
    N, num_classes, H, W = input.size()
    input = F.softmax(input, dim=1)
    arg_max = torch.argmax(input, dim=1)
    result_mean = 0
    result_freq = 0

    # start_time_cpu = time.time()
    confuse_matrix = confusion_matrix(arg_max, target, num_classes, device="cpu")
    # confuse_matrix = cm(arg_max.to("cpu"), target.to("cpu"))
    # end_time_cpu = time.time()
    # execution_time_cpu = end_time_cpu - start_time_cpu
    # print("Execution time on CPU:", execution_time_cpu)
    for i in range(num_classes):
        nii = confuse_matrix[i, i]
        # consider the case where the denominator is zero.
        if nii == 0:
            continue
        else:
            ti, tj = torch.sum(confuse_matrix[i, :]), torch.sum(confuse_matrix[:, i])
            result_mean += (nii / (ti + tj - nii))
            result_freq += (ti * nii / (ti + tj - nii))
    return result_mean / num_classes, result_freq / torch.sum(confuse_matrix)

def all_iou(input, target):
    assert len(input.size()) == 4
    assert len(target.size()) == 3
    N, num_classes, H, W = input.size()
    input = torch.softmax(input, dim=1)
    arg_max = torch.argmax(input, dim=1)
    result_iou = []
    result_mean = 0
    result_freq = 0

    confuse_matrix = confusion_matrix(arg_max, target, num_classes, device="cpu")

    for i in range(num_classes):
        nii = confuse_matrix[i, i]
        ti, tj = torch.sum(confuse_matrix[i, :]), torch.sum(confuse_matrix[:, i])

        if ti + tj - nii == 0:
            iou_value = torch.tensor(0.0)
        else:
            iou_value = nii / (ti + tj - nii)

        result_iou.append(iou_value.item())
        result_mean += iou_value
        result_freq += (ti * nii / (ti + tj - nii))

    miou = result_mean / num_classes
    fiou = result_freq / torch.sum(confuse_matrix)

    return result_iou, miou.item(), fiou.item()

def mean_iou(input, target, device='cpu'):
    """
    input: torch.FloatTensor:(N, C, H, W)
    target: torch.LongTensor:(N, H, W)
    return: Tensor
    """
    assert len(input.size()) == 4
    assert len(target.size()) == 3
    N, num_classes, H, W = input.size()
    input = F.softmax(input, dim=1)
    arg_max = torch.argmax(input, dim=1)
    result = 0

    import time

    # 记录在 CPU 上执行 confusion_matrix 函数的时间
    start_time_cpu = time.time()
    _ = confusion_matrix(arg_max, target, num_classes, device="cpu")
    end_time_cpu = time.time()
    execution_time_cpu = end_time_cpu - start_time_cpu
    print("Execution time on CPU:", execution_time_cpu)

    # 记录在 GPU 上执行 confusion_matrix 函数的时间
    start_time_gpu = time.time()
    confuse_matrix = confusion_matrix(arg_max, target, num_classes, device="gpu")
    end_time_gpu = time.time()
    execution_time_gpu = end_time_gpu - start_time_gpu

    # 输出执行时间
    print("Execution time on GPU:", execution_time_gpu)

    for i in range(num_classes):
        nii = confuse_matrix[i, i]
        # consider the case where the denominator is zero.
        if nii == 0:
            continue
        else:
            ti, tj = torch.sum(confuse_matrix[i, :]), torch.sum(confuse_matrix[:, i])
            result += (nii / (ti + tj - nii))

    return result / num_classes

def frequency_weighted_iou(input, target, device):
    """
    input: torch.FloatTensor:(N, C, H, W)
    target: torch.LongTensor:(N, H, W)
    return: Tensor
    """
    assert len(input.size()) == 4
    assert len(target.size()) == 3
    N, num_classes, H, W = input.size()
    input = F.softmax(input, dim=1)
    arg_max = torch.argmax(input, dim=1)
    # get confusion matrix
    result = 0
    confuse_matrix = confusion_matrix(arg_max, target, num_classes, device)
    for i in range(num_classes):
        nii = confuse_matrix[i, i]
        # consider the case where the denominator is zero.
        if nii == 0:
            continue
        else:
            ti, tj = torch.sum(confuse_matrix[i, :]), torch.sum(confuse_matrix[:, i])
            result += (ti * nii / (ti + tj - nii))

    return result / torch.sum(confuse_matrix)


def Metrics(input, target):
    pa = pixel_accuracy(input, target)
    # mpa = mean_pixel_accuarcy(input, target, device)
    # mi,fwi = mean_iou(input, target, device)
    #
    # return pa, mi, fwi
    # mi, fwi = mfiou(input, target)
    iou, mi, fwi = all_iou(input, target)

    return pa, mi, fwi, iou

