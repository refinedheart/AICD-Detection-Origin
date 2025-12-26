# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Loss functions."""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel
import torch.nn.functional as F
from models.common import Concat


def smooth_BCE(eps=0.1):
    """Returns label smoothing BCE targets for reducing overfitting; pos: `1.0 - 0.5*eps`, neg: `0.5*eps`. For details
    see https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441.
    """
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    """Modified BCEWithLogitsLoss to reduce missing label effects in YOLOv5 training with optional alpha smoothing."""

    def __init__(self, alpha=0.05):
        """Initializes a modified BCEWithLogitsLoss with reduced missing label effects, taking optional alpha smoothing
        parameter.
        """
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction="none")  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        """Computes modified BCE loss for YOLOv5 with reduced missing label effects, taking pred and true tensors,
        returns mean loss.
        """
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    """Applies focal loss to address class imbalance by modifying BCEWithLogitsLoss with gamma and alpha parameters."""

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        """Initializes FocalLoss with specified loss function, gamma, and alpha values; modifies loss reduction to
        'none'.
        """
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        """Calculates the focal loss between predicted and true labels using a modified BCEWithLogitsLoss."""
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    """Implements Quality Focal Loss to address class imbalance by modulating loss based on prediction confidence."""

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        """Initializes Quality Focal Loss with given loss function, gamma, alpha; modifies reduction to 'none'."""
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        """Computes the focal loss between `pred` and `true` using BCEWithLogitsLoss, adjusting for imbalance with
        `gamma` and `alpha`.
        """
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    """Computes the total loss for YOLOv5 model predictions, including classification, box, and objectness losses."""

    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False, teacher_model=None):
        """Initializes ComputeLoss with model and autobalance option, autobalances losses if True."""
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["cls_pw"]], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["obj_pw"]], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get("label_smoothing", 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h["fl_gamma"]  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance

        self.model = model

        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device

        self.teacher_model = teacher_model
        self.distill_ok = self.teacher_model is not None
        if self.distill_ok:
            self.teacher_model.float() 
            # 1. freezs teacher model
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            self.teacher_model.eval()
            de_parallel(self.teacher_model).model[-1].train()
            self.fgfi_w = h.get("distill_w", 0.005) 
            
            # 开启特征蒸馏
            self.feat_distill_enabled = True
            # YOLOv5 中间特征层：取 Detect 头前的 3 个多尺度特征层（P3、P4、P5，对应 model.model[17]、[20]、[23]，需根据 yaml 确认）
            self.student_feat_layers = [6, 8, 10]  # 学生模型的特征层索引（yolov5s.yaml 对应 C3 输出）
            self.teacher_feat_layers = [6, 8, 10]  # 教师模型的特征层索引（yolov5l.yaml 同架构，索引一致）

            dummy_img = torch.zeros(1, 3, 640, 640).to(self.device)

            with torch.no_grad():
                s_feats = self._get_intermediate_feats(self.model, dummy_img, self.student_feat_layers)
                student_channels = [f.shape[1] for f in s_feats]
                
                t_feats = self._get_intermediate_feats(self.teacher_model, dummy_img, self.teacher_feat_layers)
                teacher_channels = [f.shape[1] for f in t_feats]
            
            print(f"Distillation Channels detected: Student={student_channels}, Teacher={teacher_channels}")

            
            # 3. 传入计算好的通道数
            self.feat_projectors = self._build_feat_projectors(student_channels, teacher_channels)

    def _build_feat_projectors(self, s_channels, t_channels):
        """构建特征通道投影器（v5l 特征通道数 → v5s 特征通道数，因为 v5l 通道数是 v5s 的 2 倍）"""
        projectors = []
        
        
        for t_ch, s_ch in zip(t_channels, s_channels):
            # 1x1 卷积降维 + BatchNorm
            projector = nn.Sequential(
                nn.Conv2d(t_ch, s_ch, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(s_ch)
            ).to(self.device)
            nn.init.xavier_uniform_(projector[0].weight)
            projectors.append(projector)
        return nn.ModuleList(projectors) # 建议使用 ModuleList 以便正确注册参数

    def _get_intermediate_feats(self, model, x, layer_indices):
        """获取模型中间层特征（适配 YOLOv5 结构，跳过 Concat 等多输入模块）"""
        feats = []
        
        # 假设 Concat 模块已经被导入 (见步骤 1)
        
        for idx, m in enumerate(model.model):
            # 检查模块类型，如果是 Concat 模块，则跳过前向传播
            # 否则，Concat 模块的输入 x 此时是一个 Tensor 而非 Tensor 列表，会导致 TypeError: cat()
            if isinstance(m, Concat): 
                continue # 跳过 Concat 模块，继续下一个模块
            
            # 前向传播到当前层（只针对单输入模块：Conv, C3, SPPF 等）
            x = m(x)  
            
            # 保存特征
            if idx in layer_indices:
                feats.append(x)
                
            if idx == layer_indices[-1]:  # 到最后一个特征层后停止，提升效率
                break
        return feats

    def get_fgfi_mask(self, batch_size, feat_h, feat_w, stride, targets, anchors_layer):
        """生成 FGFI 掩码: 只在 GT 附近的 anchor 位置计算特征损失"""
        mask = torch.zeros(batch_size, 1, feat_h, feat_w, device=self.device)
        if len(targets) == 0: return mask

        # 1. 准备所有 anchor 的坐标 (Grid Scale)
        num_anchors = anchors_layer.shape[0]
        grid_y, grid_x = torch.meshgrid([torch.arange(feat_h, device=self.device), torch.arange(feat_w, device=self.device)], indexing='ij')
        grid_xy = torch.stack((grid_x, grid_y), 2).float() + 0.5
        
        # [H, W, A, 4] -> x, y, w, h
        anchors_grid = torch.cat([
            grid_xy.unsqueeze(2).expand(feat_h, feat_w, num_anchors, 2),
            (anchors_layer / stride).view(1, 1, num_anchors, 2).expand(feat_h, feat_w, num_anchors, 2)
        ], dim=-1).view(-1, 4) # Flatten [N_all, 4]

        # 转为 x1y1x2y2
        anchors_xyxy = torch.zeros_like(anchors_grid)
        anchors_xyxy[:, 0], anchors_xyxy[:, 1] = anchors_grid[:, 0] - anchors_grid[:, 2]/2, anchors_grid[:, 1] - anchors_grid[:, 3]/2
        anchors_xyxy[:, 2], anchors_xyxy[:, 3] = anchors_grid[:, 0] + anchors_grid[:, 2]/2, anchors_grid[:, 1] + anchors_grid[:, 3]/2

        # 2. 遍历 Batch 计算 Mask
        for b_idx in range(batch_size):
            t = targets[targets[:, 0] == b_idx]
            if len(t) == 0: continue
            
            # GT 转为 grid scale
            gt_box = t[:, 2:6] * torch.tensor([feat_w, feat_h, feat_w, feat_h], device=self.device)
            gt_xyxy = torch.zeros_like(gt_box)
            gt_xyxy[:, 0], gt_xyxy[:, 1] = gt_box[:, 0] - gt_box[:, 2]/2, gt_box[:, 1] - gt_box[:, 3]/2
            gt_xyxy[:, 2], gt_xyxy[:, 3] = gt_box[:, 0] + gt_box[:, 2]/2, gt_box[:, 1] + gt_box[:, 3]/2

            # 计算 IoU: [N_anchors, N_gt]
            ious = bbox_iou(anchors_xyxy.unsqueeze(1), gt_xyxy.unsqueeze(0), x1y1x2y2=True)
            
            # FGFI 阈值筛选: IoU > 0.5 * Max_IoU
            max_iou_per_gt, _ = ious.max(dim=0)
            thresholds = max_iou_per_gt * 0.5
            active_anchors = (ious > thresholds.unsqueeze(0)).any(dim=1).float() # [N_anchors]
            
            # 还原为空间 Mask [H, W] (只要该位置有任意 anchor 被激活，该像素点就算激活)
            mask[b_idx, 0] = active_anchors.view(feat_h, feat_w, num_anchors).max(dim=2)[0]
            
        return mask

    def __call__(self, p, targets, imgs=None):  # predictions, targets
        """Performs forward pass, calculating class, box, and object loss for given predictions and targets."""
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        ldistill = torch.zeros(1, device=self.device) 
        lfeat_distill = torch.zeros(1, device=self.device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            if n := b.shape[0]:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()
                # distill loss（只在训练阶段做；验证阶段直接跳过）
        
        
        # FGFI 蒸馏逻辑
        if self.distill_ok and self.model.training and imgs is not None:
            imgs_fp32 = imgs.float()
            with torch.no_grad():
                 # 只提取 Teacher 特征，不再需要 Teacher 的预测输出 (p)
                teacher_feats = self._get_intermediate_feats(self.teacher_model, imgs_fp32, self.teacher_feat_layers)
            
            # 提取 Student 特征
            student_feats = self._get_intermediate_feats(de_parallel(self.model), imgs_fp32, self.student_feat_layers)

            # 计算 FGFI Loss
            if len(student_feats) == len(teacher_feats):
                for i, (s_feat, t_feat, projector) in enumerate(zip(student_feats, teacher_feats, self.feat_projectors)):
                    # 1. 特征对齐
                    t_feat_proj = projector(t_feat)
                    if s_feat.shape[2:] != t_feat_proj.shape[2:]:
                         t_feat_proj = F.interpolate(t_feat_proj, size=s_feat.shape[2:], mode="bilinear", align_corners=False)
                    
                    # 2. 生成 Mask (基于当前层的 stride 和 anchors)
                    B, C, H, W = s_feat.shape
                    stride = imgs.shape[-1] / W
                    mask = self.get_fgfi_mask(B, H, W, stride, targets, self.anchors[i])
                    
                    # 3. 计算 Masked MSE Loss
                    diff = (s_feat - t_feat_proj) ** 2
                    masked_diff = diff * mask
                    # 归一化: 除以 Mask 中的正样本像素数，防止 loss 过大
                    num_pos = mask.sum() * C + 1e-6 
                    lfeat_distill += masked_diff.sum() / num_pos

            # 乘以 FGFI 权重
            lfeat_distill *= self.fgfi_w
        
        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp["box"]
        lobj *= self.hyp["obj"]
        lcls *= self.hyp["cls"]
        bs = tobj.shape[0]  # batch size

        total_loss = (lbox + lobj + lcls + ldistill + lfeat_distill) * bs

        if imgs is not None:
            return total_loss, torch.cat((lbox, lobj, lcls, ldistill)).detach()
        else:
            return total_loss, torch.cat((lbox, lobj, lcls)).detach()

        # return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        """Prepares model targets from input targets (image,class,x,y,w,h) for loss computation, returning class, box,
        indices, and anchors.
        """
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = (
            torch.tensor(
                [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [-1, 0],
                    [0, -1],  # j,k,l,m
                    # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                ],
                device=self.device,
            ).float()
            * g
        )  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp["anchor_t"]  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
