import numpy as np
from numpy import ndarray as arr
from typing import Union, Tuple, List

def opinion_to_proba(opinion: arr) -> arr:

    method = 1
    if method == 1:
        proba = opinion[:, :-1].copy()
        proba /= np.sum(proba, axis=1)[:, np.newaxis]
    elif method == 2:
        proba = opinion[:, :-1].copy()
        proba += opinion[:, -1] / (opinion.shape[1] - 1)
    elif method == 3:
        proba = opinion[:, :-1].copy()
    else:
        raise ValueError("method must be 1, 2 or 3!")
    if np.sum(np.isnan(proba)) != 0:
        np.nan_to_num(proba, 0)
    return proba


def joint_multi_opinion(opinion_list: List[arr], method: int = 2) -> arr:


    assert len(opinion_list) > 1, "The opinion_list parameter should contain at least two opinions"
    if method == 1:
        joint_func = joint_opinions_1
    elif method == 2:
        joint_func = joint_opinions_2

    joint_opinion = opinion_list[0]
    for opinion in opinion_list[1:]:
        joint_opinion = joint_func(joint_opinion, opinion)

    return joint_opinion

def joint_opinions_1(opinion1: arr, opinion2: arr) -> arr:

    combining_B = np.array(
        [np.outer(opinion1[i], opinion2[i]) for i in range(opinion1.shape[0])])
    joint_opinion = np.empty_like(opinion1)

    mask = np.ones_like(combining_B[0], dtype=bool)  # 创建一个与mat形状相同的掩码数组，并将所有元素设置为True
    mask[:, -1] = False  # 将最后一行元素的掩码设置为False
    mask[-1, :] = False  # 将最后一列元素的掩码设置为False
    mask[np.diag_indices_from(mask)] = False  # 将对角线元素的掩码设置为False
    C = np.array([np.sum(mat[mask]) for mat in combining_B])  # 计算C

    scale_factor = 1 / (1 - C)
    for i in range(joint_opinion.shape[1] - 1):
        joint_opinion[:, i] = scale_factor * (combining_B[:, i, i] + combining_B[:, -1, i] + combining_B[:, i, -1])
    joint_opinion[:, -1] = scale_factor * combining_B[:, -1, -1]
    return joint_opinion


def joint_opinions_2(opinion1: arr, opinion2: arr):

    combining_B = np.array(
        [np.outer(opinion1[i], opinion2[i]).T for i in
         range(opinion1.shape[0])])  # shape of (#samples, #classes+1, #classes+1)
    joint_opinion = np.empty_like(opinion1)

    mask = np.ones_like(combining_B[0], dtype=bool)  # 创建一个与mat形状相同的掩码数组，并将所有元素设置为True
    mask[:, -1] = False  # 将最后一行元素的掩码设置为False
    mask[-1, :] = False  # 将最后一列元素的掩码设置为False
    mask[np.diag_indices_from(mask)] = False  # 将对角线元素的掩码设置为False
    C = np.array([np.sum(mat[mask]) for mat in combining_B])  # 计算C, shape of (#samples, )

    # 预测结果
    y_pred_1 = np.argmax(opinion1[:, :-1], axis=1)
    y_pred_2 = np.argmax(opinion2[:, :-1], axis=1)

    u1 = opinion1[:, -1]
    u2 = opinion2[:, -1]
    u_mul = u1 * u2
    u_mul_signed = np.where(y_pred_1 == y_pred_2, -u_mul, u_mul)
    u_mean = np.mean([u1, u2], axis=0)

    delta = 1 - C - u_mul + u_mean

    scale_factor = 1 / (delta + u_mul_signed)

    # 计算联合belief, 遍历每个类
    for i in range(joint_opinion.shape[1] - 1):
        joint_opinion[:, i] = scale_factor * (combining_B[:, i, i] + combining_B[:, -1, i] + combining_B[:, i, -1])
    # 计算联合uncertainty
    joint_opinion[:, -1] = scale_factor * (u_mul_signed + u_mean)

    return joint_opinion