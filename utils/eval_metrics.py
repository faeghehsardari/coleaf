import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


def Precision(X_pre, X_gt):
    N = len(X_pre)
    p = 0.0
    for i in range(N):
        x = X_pre[i, :]
        y = X_gt[i, :]
        p += np.sum(x * y) / np.sum(x)
    return p / N


def Recall(X_pre, X_gt):
    N = len(X_pre)
    p = 0.0
    for i in range(N):
        x = X_pre[i, :]
        y = X_gt[i, :]
        p += np.sum(x * y) / np.sum(y)
    return p / N


def F1(X_pre, X_gt):
    N = len(X_pre)
    p = 0
    for i in range(N):
        x = X_pre[i, :]
        y = X_gt[i, :]
        p += 2 * np.sum(x * y) / (np.sum(x) + np.sum(y))
    return p / N


def event_level(SO_a, SO_v, SO_av, GT_a, GT_v, GT_av):
    # extract events
    N = 25
    event_p_a = [None for n in range(25)]
    event_gt_a = [None for n in range(25)]
    event_p_v = [None for n in range(25)]
    event_gt_v = [None for n in range(25)]
    event_p_av = [None for n in range(25)]
    event_gt_av = [None for n in range(25)]

    TP_a = np.zeros(25)
    TP_v = np.zeros(25)
    TP_av = np.zeros(25)

    FP_a = np.zeros(25)
    FP_v = np.zeros(25)
    FP_av = np.zeros(25)

    FN_a = np.zeros(25)
    FN_v = np.zeros(25)
    FN_av = np.zeros(25)

    for n in range(N):
        seq_pred = SO_a[n, :]
        if np.sum(seq_pred) != 0:
            x = extract_event(seq_pred, n)
            event_p_a[n] = x
        seq_gt = GT_a[n, :]
        if np.sum(seq_gt) != 0:
            x = extract_event(seq_gt, n)
            event_gt_a[n] = x

        seq_pred = SO_v[n, :]
        if np.sum(seq_pred) != 0:
            x = extract_event(seq_pred, n)
            event_p_v[n] = x
        seq_gt = GT_v[n, :]
        if np.sum(seq_gt) != 0:
            x = extract_event(seq_gt, n)
            event_gt_v[n] = x

        seq_pred = SO_av[n, :]
        if np.sum(seq_pred) != 0:
            x = extract_event(seq_pred, n)
            event_p_av[n] = x
        seq_gt = GT_av[n, :]
        if np.sum(seq_gt) != 0:
            x = extract_event(seq_gt, n)
            event_gt_av[n] = x

        tp, fp, fn = event_wise_metric(event_p_a[n], event_gt_a[n])
        TP_a[n] += tp
        FP_a[n] += fp
        FN_a[n] += fn

        tp, fp, fn = event_wise_metric(event_p_v[n], event_gt_v[n])
        TP_v[n] += tp
        FP_v[n] += fp
        FN_v[n] += fn

        tp, fp, fn = event_wise_metric(event_p_av[n], event_gt_av[n])
        TP_av[n] += tp
        FP_av[n] += fp
        FN_av[n] += fn

    TP = TP_a + TP_v
    FN = FN_a + FN_v
    FP = FP_a + FP_v

    n = len(FP_a)
    F_a = []
    for ii in range(n):
        if (TP_a + FP_a)[ii] != 0 or (TP_a + FN_a)[ii] != 0:
            F_a.append(2 * TP_a[ii] / (2 * TP_a[ii] + (FN_a + FP_a)[ii]))

    F_v = []
    for ii in range(n):
        if (TP_v + FP_v)[ii] != 0 or (TP_v + FN_v)[ii] != 0:
            F_v.append(2 * TP_v[ii] / (2 * TP_v[ii] + (FN_v + FP_v)[ii]))

    F = []
    for ii in range(n):
        if (TP + FP)[ii] != 0 or (TP + FN)[ii] != 0:
            F.append(2 * TP[ii] / (2 * TP[ii] + (FN + FP)[ii]))

    F_av = []
    for ii in range(n):
        if (TP_av + FP_av)[ii] != 0 or (TP_av + FN_av)[ii] != 0:
            F_av.append(2 * TP_av[ii] / (2 * TP_av[ii] + (FN_av + FP_av)[ii]))

    if len(F_a) == 0:
        f_a = 1.0
    else:
        f_a = (sum(F_a) / len(F_a))

    if len(F_v) == 0:
        f_v = 1.0
    else:
        f_v = (sum(F_v) / len(F_v))

    if len(F) == 0:
        f = 1.0
    else:
        f = (sum(F) / len(F))
    if len(F_av) == 0:
        f_av = 1.0
    else:
        f_av = (sum(F_av) / len(F_av))

    return f_a, f_v, f, f_av


def segment_level(SO_a, SO_v, SO_av, GT_a, GT_v, GT_av):
    # compute F scores
    TP_a = np.sum(SO_a * GT_a, axis=1)
    FN_a = np.sum((1 - SO_a) * GT_a, axis=1)
    FP_a = np.sum(SO_a * (1 - GT_a), axis=1)

    n = len(FP_a)
    F_a = []
    for ii in range(n):
        if (TP_a + FP_a)[ii] != 0 or (TP_a + FN_a)[ii] != 0:
            F_a.append(2 * TP_a[ii] / (2 * TP_a[ii] + (FN_a + FP_a)[ii]))

    # visual
    TP_v = np.sum(SO_v * GT_v, axis=1)
    FN_v = np.sum((1 - SO_v) * GT_v, axis=1)
    FP_v = np.sum(SO_v * (1 - GT_v), axis=1)
    F_v = []
    for ii in range(n):
        if (TP_v + FP_v)[ii] != 0 or (TP_v + FN_v)[ii] != 0:
            F_v.append(2 * TP_v[ii] / (2 * TP_v[ii] + (FN_v + FP_v)[ii]))

    TP = TP_a + TP_v
    FN = FN_a + FN_v
    FP = FP_a + FP_v

    n = len(FP)

    F = []
    for ii in range(n):
        if (TP + FP)[ii] != 0 or (TP + FN)[ii] != 0:
            F.append(2 * TP[ii] / (2 * TP[ii] + (FN + FP)[ii]))

    TP_av = np.sum(SO_av * GT_av, axis=1)
    FN_av = np.sum((1 - SO_av) * GT_av, axis=1)
    FP_av = np.sum(SO_av * (1 - GT_av), axis=1)
    n = len(FP_av)
    F_av = []
    for ii in range(n):
        if (TP_av + FP_av)[ii] != 0 or (TP_av + FN_av)[ii] != 0:
            F_av.append(2 * TP_av[ii] / (2 * TP_av[ii] + (FN_av + FP_av)[ii]))

    if len(F_a) == 0:
        f_a = 1.0
    else:
        f_a = (sum(F_a) / len(F_a))

    if len(F_v) == 0:
        f_v = 1.0
    else:
        f_v = (sum(F_v) / len(F_v))

    if len(F) == 0:
        f = 1.0
    else:
        f = (sum(F) / len(F))
    if len(F_av) == 0:
        f_av = 1.0
    else:
        f_av = (sum(F_av) / len(F_av))

    return f_a, f_v, f, f_av


def to_vec(start, end):
    x = np.zeros(10)
    for i in range(start, end):
        x[i] = 1
    return x


def extract_event(seq, n):
    x = []
    i = 0
    while i < 10:
        if seq[i] == 1:
            start = i
            if i + 1 == 10:
                i = i + 1
                end = i
                x.append(to_vec(start, end))
                break

            for j in range(i + 1, 10):
                if seq[j] != 1:
                    i = j + 1
                    end = j
                    x.append(to_vec(start, end))
                    break
                else:
                    i = j + 1
                    if i == 10:
                        end = i
                        x.append(to_vec(start, end))
                        break
        else:
            i += 1
    return x


def event_wise_metric(event_p, event_gt):
    TP = 0
    FP = 0
    FN = 0

    if event_p is not None:
        num_event = len(event_p)
        for i in range(num_event):
            x1 = event_p[i]
            if event_gt is not None:
                nn = len(event_gt)
                flag = True
                for j in range(nn):
                    x2 = event_gt[j]
                    if np.sum(x1 * x2) >= 0.5 * np.sum(x1 + x2 - x1 * x2):
                        TP += 1
                        flag = False
                        break
                if flag:
                    FP += 1
            else:
                FP += 1

    if event_gt is not None:
        num_event = len(event_gt)
        for i in range(num_event):
            x1 = event_gt[i]
            if event_p is not None:
                nn = len(event_p)
                flag = True
                for j in range(nn):
                    x2 = event_p[j]
                    if np.sum(x1 * x2) >= 0.5 * np.sum(x1 + x2 - x1 * x2):
                        flag = False
                        break
                if flag:
                    FN += 1
            else:
                FN += 1
    return TP, FP, FN


def print_overall_metric(F_seg_a, F_seg_v, F_seg, F_seg_av, F_event_a, F_event_v, F_event, F_event_av, verbose=True):
    audio_segment_level = 100 * np.mean(np.array(F_seg_a))
    visual_segment_level = 100 * np.mean(np.array(F_seg_v))
    av_segment_level = 100 * np.mean(np.array(F_seg_av))

    avg_type_seg = (100 * np.mean(np.array(F_seg_av)) +
                    100 * np.mean(np.array(F_seg_a)) +
                    100 * np.mean(np.array(F_seg_v))) / 3.
    avg_event_seg = 100 * np.mean(np.array(F_seg))

    audio_event_level = 100 * np.mean(np.array(F_event_a))
    visual_event_level = 100 * np.mean(np.array(F_event_v))
    av_event_level = 100 * np.mean(np.array(F_event_av))

    avg_type_eve = (100 * np.mean(np.array(F_event_av)) +
                    100 * np.mean(np.array(F_event_a)) +
                    100 * np.mean(np.array(F_event_v))) / 3.
    avg_event_eve = 100 * np.mean(np.array(F_event))

    if verbose:
        print('{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.3f}'.
              format(audio_segment_level, visual_segment_level, av_segment_level, avg_type_seg, avg_event_seg,
                     audio_event_level, visual_event_level, av_event_level, avg_type_eve, avg_event_eve,
                     (audio_segment_level + visual_segment_level + av_segment_level + avg_type_seg + avg_event_seg +
                     audio_event_level + visual_event_level + av_event_level + avg_type_eve + avg_event_eve) / 10.0))

    return audio_segment_level, visual_segment_level, av_segment_level, avg_type_seg, avg_event_seg, \
           audio_event_level, visual_event_level, av_event_level, avg_type_eve, avg_event_eve, \
           (audio_segment_level + visual_segment_level + av_segment_level + avg_type_seg + avg_event_seg +
            audio_event_level + visual_event_level + av_event_level + avg_type_eve + avg_event_eve) / 10.0


def KL_MC_loss(pred, target):
    b, c = pred.shape

    new_pred = torch.zeros(b, c, 2).to(pred.device)
    new_target = torch.zeros(b, c, 2).to(target.device)
    lk_loss= torch.nn.KLDivLoss(reduction="batchmean")
    loss = 0
    for i in range(c):
        new_pred[:, i, 0] = pred[:, i]
        new_pred[:, i, 1] = 1 - pred[:, i]

        new_pred[:, i, :] = torch.log(new_pred[:, i, :])

        new_target[:, i, 0] = target[:, i]
        new_target[:, i, 1] = 1 - target[:, i]

        loss += lk_loss(torch.log(pred[:, i]), target[:, i])

    loss = loss / c
    return loss


def KL_MC_loss2(pred, target):
    b, t, c = pred.shape

    new_pred = torch.zeros(b, t, c, 2).to(pred.device)
    new_target = torch.zeros(b, t, c, 2).to(target.device)
    lk_loss= torch.nn.KLDivLoss(reduction="batchmean")
    loss = 0
    for i in range(c):
        new_pred[:, :, i, 0] = pred[:, :, i]
        new_pred[:, :, i, 1] = 1 - pred[:, :, i]

        new_pred[:, :, i, :] = torch.log(new_pred[:, :, i, :])

        new_target[:, :, i, 0] = target[:, :, i]
        new_target[:, :, i, 1] = 1 - target[:, :, i]

        loss += lk_loss(torch.log(pred[:, :, i]), target[:, :, i])

    loss = loss / c
    return loss

def KL_MC_loss3(pred, target, threshold):
    b, c = pred.shape

    new_pred = torch.zeros(b, c, 2).to(pred.device)
    pred2 = torch.zeros(b, c).to(pred.device)
    new_target = torch.zeros(b, c, 2).to(target.device)
    lk_loss= torch.nn.KLDivLoss(reduction="batchmean")
    for i in range(b):
        for j in range(c):
            if pred[i, j] > threshold:
                pred2[i, j] = target[i, j]
            else:
                pred2[i,j] = pred[i, j]
    loss = 0
    for i in range(c):
        new_pred[:, i, 0] = pred2[:, i]
        new_pred[:, i, 1] = 1 - pred2[:, i]

        new_pred[:, i, :] = torch.log(new_pred[:, i, :])

        new_target[:, i, 0] = target[:, i]
        new_target[:, i, 1] = 1 - target[:, i]


        loss += lk_loss(torch.log(pred2[:, i]), target[:, i])

    loss = loss / c
    return loss
class AsymmetricLoss(nn.Module):
    # def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
    def __init__(self, gamma_neg=1, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        # x_sigmoid = torch.sigmoid(x)
        # xs_pos = x_sigmoid
        # xs_neg = 1 - x_sigmoid

        xs_pos = x
        xs_neg = 1 - x

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()

class FocalLoss2d(nn.modules.loss._WeightedLoss):

    def __init__(self, gamma=1, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='sum', balance_param=0.25):
        super(FocalLoss2d, self).__init__(weight, size_average, reduce, reduction)
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.balance_param = balance_param

    def forward(self, input, target):
        # inputs and targets are assumed to be BatchxClasses
        assert len(input.shape) == len(target.shape)
        assert input.size(0) == target.size(0)
        assert input.size(1) == target.size(1)

        weight = Variable(self.weight)

        # compute the negative likelyhood
        # logpt = - F.binary_cross_entropy_with_logits(input, target, pos_weight=weight, reduction=self.reduction)
        logpt = - F.binary_cross_entropy_with_logits(input, target, reduction=self.reduction)
        pt = torch.exp(logpt)

        # compute the loss
        focal_loss = -((1 - pt) ** self.gamma) * logpt
        balanced_focal_loss = self.balance_param * focal_loss
        return balanced_focal_loss


def cc_loss(logits_student, logits_teacher, mode=1, reduce=True):
    batch_size, class_num = logits_teacher.shape

    loss_l1= nn.L1Loss()

    student_matrix = logits_student.unsqueeze(dim=2) @ logits_student.unsqueeze(dim=1)
    teacher_matrix = logits_teacher.unsqueeze(dim=2) @ logits_teacher.unsqueeze(dim=1)


    if reduce:
        if mode == 1:
            consistency_loss = ((teacher_matrix - student_matrix) ** 2).sum() / class_num * batch_size
        else:
            consistency_loss = loss_l1(student_matrix, teacher_matrix)
    else:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2) / class_num * batch_size

    return consistency_loss


def cc_loss2(logits_student, logits_teacher, reduce= True):
    batch_size, temporal_dim, class_num = logits_teacher.shape

    student_matrix = logits_student.unsqueeze(dim=3) @ logits_student.unsqueeze(dim=2)
    teacher_matrix = logits_teacher.unsqueeze(dim=3) @ logits_teacher.unsqueeze(dim=2)

    if reduce:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2).sum() / class_num * batch_size * temporal_dim
    else:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2) / class_num * batch_size * temporal_dim

    return consistency_loss





