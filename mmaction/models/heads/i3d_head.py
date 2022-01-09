import torch.nn as nn
import torch.nn.functional as F
import torch
from mmcv.cnn import normal_init
from abc import ABCMeta

from ..builder import HEADS
from .base import BaseHead, BaseHeadZeroShot


@HEADS.register_module()
class I3DHead(BaseHead):
    """Classification head for I3D.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 spatial_type='avg',
                 dropout_ratio=0.5,
                 init_std=0.01,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = None

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [N, in_channels, 4, 7, 7]
        if self.avg_pool is not None:
            x = self.avg_pool(x)
        # [N, in_channels, 1, 1, 1]
        if self.dropout is not None:
            x = self.dropout(x)
        # [N, in_channels, 1, 1, 1]
        x = x.view(x.shape[0], -1)
        # [N, in_channels]
        cls_score = self.fc_cls(x)
        # [N, num_classes]
        return cls_score

@HEADS.register_module()
class GlobalPointerHeadZeroShot(nn.Module, metaclass=ABCMeta):

    def __init__(self, 
        nb_in_feature_channel=768,
        use_bias=True,
        RoPE=False,
        loss_cfg=dict(
            type='multilabel_sjl'
        ),
        **kwargs):
        super().__init__()

        self.nb_in_feature_channel = nb_in_feature_channel
        self.q_layer = nn.Linear(nb_in_feature_channel, nb_in_feature_channel, use_bias)
        self.k_layer = nn.Linear(nb_in_feature_channel, nb_in_feature_channel, use_bias)
        self.q_layer_cls = nn.Linear(nb_in_feature_channel, nb_in_feature_channel, use_bias)
        self.k_layer_cls = nn.Linear(nb_in_feature_channel, nb_in_feature_channel, use_bias)
        self.RoPE = RoPE
        self.loss_cfg = loss_cfg
        self.act_sigmoid = nn.Sigmoid()
        self.bce_loss = nn.BCELoss(reduction='none')

    def init_weights(self):
        """Initiate the parameters from scratch."""
        # normal_init(self.fc_cls, std=self.init_std)
        pass


    def sinusoidal_position_embedding(self, inputs):
        bs, N, C = inputs.shape
        device = inputs.device

        position_ids = torch.arange(0, N, dtype=torch.float)[..., None]
        indices = torch.arange(0, C // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / C)

        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.reshape([N, C])
        pos_embeddings_batch = embeddings[None]

        return pos_embeddings_batch.to(device)  # [1, N, C]

    def global_pointer(self, inputs, q_layer, k_layer):
        x = inputs['feature']  # [bs, N, 768]
        N = x.shape[1]

        x_q = q_layer(x)
        x_k = k_layer(x)
        if self.RoPE:
            pos_emb = self.sinusoidal_position_embedding(x)
            cos_pos = pos_emb[..., 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)

            x_q2 = torch.stack([-x_q[..., 1::2], x_q[...,::2]], -1)
            x_q2 = x_q2.reshape(x.shape)
            x_q = x_q * cos_pos + x_q2 * sin_pos

            x_k2 = torch.stack([-x_k[..., 1::2], x_k[...,::2]], -1)
            x_k2 = x_k2.reshape(x.shape)
            x_k = x_k * cos_pos + x_k2 * sin_pos

        # qk计算内积
        # logits = x_q @ x_k.permute(0, 2, 1)
        logits = torch.matmul(x_q, x_k.transpose(-1, -2))
        mask = torch.tril(torch.ones(N, N), diagonal=-1).to(x.device)  # 下三角矩阵
        logits = logits - mask * 1e10
        return logits / (self.nb_in_feature_channel**0.5)

    def forward(self, inputs):
        global_pointer_output_1 = self.global_pointer(inputs, self.q_layer, self.k_layer)
        global_pointer_output_2 = self.global_pointer(inputs, self.q_layer_cls, self.k_layer_cls)
        output = dict(global_pointer=global_pointer_output_1, global_pointer_cls=global_pointer_output_2)
        return output

    # from 苏神
    def multilabel_categorical_crossentropy(self, y_true, y_pred):
        """多标签分类的交叉熵
        说明：y_true和y_pred的shape一致，y_true的元素非0即1，
            1表示对应的类为目标类，0表示对应的类为非目标类。
        警告：请保证y_pred的值域是全体实数，换言之一般情况下y_pred
            不用加激活函数，尤其是不能加sigmoid或者softmax！预测
            阶段则输出y_pred大于0的类。如有疑问，请仔细阅读并理解
            本文。
        """
        # torch
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 1e12
        y_pred_pos = y_pred - (1 - y_true) * 1e12
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        loss = neg_loss + pos_loss
        return loss


    # from 苏神
    def global_pointer_crossentropy(self, y_true, y_pred):
        """给GlobalPointer设计的交叉熵
        """
        y_pred = y_pred['global_pointer_cls']
        gt_iou_map = y_true

        y_true = (gt_iou_map > 0.9).float()

        bs, N, N = y_pred.shape
        y_true = y_true.reshape([bs, -1])
        y_pred = y_pred.reshape([bs, -1])
        return torch.mean(self.multilabel_categorical_crossentropy(y_true, y_pred))


    def bmn_mse(self, y_true, y_pred, high_temporal_iou_threshold=0.7, low_temporal_iou_threshold=0.3):
        y_pred = y_pred['global_pointer']

        y_pred_sigmoid = torch.sigmoid(y_pred)
        bs, N, N = y_pred.shape
        mask = torch.tril(torch.ones(N, N), diagonal=-1).to(y_pred.device)  # 下三角矩阵
        mask = (1-mask).float()

        gt_iou_map = y_true.float()
        pred_score = y_pred_sigmoid.float()

        u_hmask = (gt_iou_map > high_temporal_iou_threshold).float()
        u_mmask = ((gt_iou_map <= high_temporal_iou_threshold) & (gt_iou_map > low_temporal_iou_threshold)).float()
        u_lmask = ((gt_iou_map <= low_temporal_iou_threshold) & (gt_iou_map > 0.)).float()
        u_lmask = u_lmask * mask

        num_h = torch.sum(u_hmask)
        num_m = torch.sum(u_mmask)
        num_l = torch.sum(u_lmask)

        r_m = num_h / (num_m+1e-10)
        u_smmask = torch.rand_like(gt_iou_map)
        u_smmask = u_mmask * u_smmask
        u_smmask = (u_smmask > (1. - r_m)).float()

        r_l = num_h / (num_l+1e-10)
        u_slmask = torch.rand_like(gt_iou_map)
        u_slmask = u_lmask * u_slmask
        u_slmask = (u_slmask > (1. - r_l)).float()

        weights = u_hmask + u_smmask + u_slmask

        loss = F.mse_loss(pred_score * weights, gt_iou_map * weights)
        loss = torch.sum(loss * torch.ones_like(weights)) / torch.sum(weights)
        return loss*10

    def faster_rcnn_cls_loss(self, y_true, y_pred, high_temporal_iou_threshold=0.7, low_temporal_iou_threshold=0.3):
        y_pred = y_pred['global_pointer_cls']

        y_pred_sigmoid = torch.sigmoid(y_pred)
        bs, N, N = y_pred.shape
        mask = torch.tril(torch.ones(N, N), diagonal=-1).to(y_pred.device)  # 下三角矩阵
        mask = (1-mask).float()

        gt_iou_map = y_true.float()
        pred_score = y_pred_sigmoid.float()
        h_label = torch.ones_like(gt_iou_map)
        s_label = torch.zeros_like(gt_iou_map)

        u_hmask = (gt_iou_map > high_temporal_iou_threshold).float()
        u_lmask = ((gt_iou_map <= low_temporal_iou_threshold) & (gt_iou_map > 0.)).float()
        u_lmask = u_lmask * mask

        num_h = torch.sum(u_hmask)
        num_l = torch.sum(u_lmask)

        r_l = num_h / (num_l+1e-10)
        u_slmask = torch.rand_like(gt_iou_map)
        u_slmask = u_lmask * u_slmask
        u_slmask = (u_slmask > (1. - r_l)).float()

        h_cls_loss = self.bce_loss(pred_score*u_hmask, h_label*u_hmask)
        s_cls_loss = self.bce_loss(pred_score*u_slmask, s_label*u_slmask)
        h_cls_loss = torch.sum(h_cls_loss*u_hmask)/(torch.sum(u_hmask)+1e-10)
        s_cls_loss = torch.sum(s_cls_loss*u_slmask)/(torch.sum(u_slmask)+1e-10)
        loss = h_cls_loss + s_cls_loss
        return loss

    def bmn_cls_loss(self, y_true, y_pred, threshold=0.9, ratio_range=(1.05, 21), eps=1e-10):
        y_pred = y_pred['global_pointer_cls']

        y_pred_sigmoid = torch.sigmoid(y_pred)
        bs, N, N = y_pred.shape
        mask = torch.tril(torch.ones(N, N), diagonal=-1).to(y_pred.device)  # 下三角矩阵
        mask = (1-mask).float()

        gt_iou_map = y_true.float()
        pred_score = y_pred_sigmoid.float()

        pmask = (gt_iou_map > threshold).float()
        nmask = (gt_iou_map <= threshold).float()
        nmask = nmask * mask

        num_positive = max(torch.sum(pmask), 1)
        num_entries = num_positive + torch.sum(nmask)
        ratio = num_entries / num_positive
        ratio = torch.clamp(ratio, ratio_range[0], ratio_range[1])

        coef_0 = 0.5 * ratio / (ratio - 1)
        coef_1 = 0.5 * ratio

        loss_pos = coef_1 * torch.log(pred_score + eps) * pmask
        loss_neg = coef_0 * torch.log(1.0 - pred_score + eps) * nmask
        loss = -1 * torch.sum(loss_pos + loss_neg) / num_entries
        return loss

    # def bmn_pem_cls_loss(self, y_true, y_pred, threshold=0.9, )

    def loss(self, global_pointer_output, gt_labels, **kwargs):
        # global_pointer.shape = gt_labels.shape = [bs, N, N]

        if self.loss_cfg['type'] == 'multilabel_sjl':
            loss_cls = self.global_pointer_crossentropy(gt_labels, global_pointer_output)
        if self.loss_cfg['type'] == 'bmn_mse':
            loss_reg = self.bmn_mse(gt_labels, global_pointer_output)
            # loss_cls = self.faster_rcnn_cls_loss(gt_labels, global_pointer_output)
            loss_cls = self.bmn_cls_loss(gt_labels, global_pointer_output)
            losses = dict(loss_cls=loss_cls, loss_reg=loss_reg)
            return losses
        if self.loss_cfg['type'] == 'multilabel_sjl_bmn_mse':
            loss_reg = self.bmn_mse(gt_labels, global_pointer_output)
            loss_cls = self.global_pointer_crossentropy(gt_labels, global_pointer_output)
            losses = dict(loss_cls=loss_cls, loss_reg=loss_reg)
            return losses
        else:
            loss_cls = torch.mean(global_pointer_output - gt_labels)

        losses = dict(loss_cls=loss_cls)
        return losses
