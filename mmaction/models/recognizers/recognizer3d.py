import torch
from torch import nn

from ..builder import RECOGNIZERS
from .base import BaseRecognizer, BaseRecognizerZeroShot
from mmcv.runner import auto_fp16
import numpy as np
from ..localizers.utils import post_processing

from rich import print
from mmaction.localization import soft_nms
import torch.nn.functional as F

@RECOGNIZERS.register_module()
class Recognizer3D(BaseRecognizer):
    """3D recognizer model framework."""

    def forward_train(self, imgs, labels, **kwargs):
        """Defines the computation performed at every call when training."""

        assert self.with_cls_head
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        losses = dict()

        x = self.extract_feat(imgs)
        if self.with_neck:
            x, loss_aux = self.neck(x, labels.squeeze())
            losses.update(loss_aux)

        cls_score = self.cls_head(x)
        gt_labels = labels.squeeze()
        loss_cls = self.cls_head.loss(cls_score, gt_labels, **kwargs)
        losses.update(loss_cls)

        return losses

    def _do_test(self, imgs):
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""
        batches = imgs.shape[0]
        num_segs = imgs.shape[1]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])

        if self.max_testing_views is not None:
            total_views = imgs.shape[0]
            assert num_segs == total_views, (
                'max_testing_views is only compatible '
                'with batch_size == 1')
            view_ptr = 0
            feats = []
            while view_ptr < total_views:
                batch_imgs = imgs[view_ptr:view_ptr + self.max_testing_views]
                x = self.extract_feat(batch_imgs)
                if self.with_neck:
                    x, _ = self.neck(x)
                feats.append(x)
                view_ptr += self.max_testing_views
            # should consider the case that feat is a tuple
            if isinstance(feats[0], tuple):
                len_tuple = len(feats[0])
                feat = [
                    torch.cat([x[i] for x in feats]) for i in range(len_tuple)
                ]
                feat = tuple(feat)
            else:
                feat = torch.cat(feats)
        else:
            feat = self.extract_feat(imgs)
            if self.with_neck:
                feat, _ = self.neck(feat)

        if self.feature_extraction:
            # perform spatio-temporal pooling
            avg_pool = nn.AdaptiveAvgPool3d(1)
            if isinstance(feat, tuple):
                feat = [avg_pool(x) for x in feat]
                # concat them
                feat = torch.cat(feat, axis=1)
            else:
                feat = avg_pool(feat)
            # squeeze dimensions
            feat = feat.reshape((batches, num_segs, -1))
            # temporal average pooling
            feat = feat.mean(axis=1)
            return feat

        # should have cls_head if not extracting features
        assert self.with_cls_head
        cls_score = self.cls_head(feat)
        cls_score = self.average_clip(cls_score, num_segs)
        return cls_score

    def forward_test(self, imgs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        return self._do_test(imgs).cpu().numpy()

    def forward_dummy(self, imgs, softmax=False):
        """Used for computing network FLOPs.

        See ``tools/analysis/get_flops.py``.

        Args:
            imgs (torch.Tensor): Input images.

        Returns:
            Tensor: Class score.
        """
        assert self.with_cls_head
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        x = self.extract_feat(imgs)

        if self.with_neck:
            x, _ = self.neck(x)

        outs = self.cls_head(x)
        if softmax:
            outs = nn.functional.softmax(outs)
        return (outs, )

    def forward_gradcam(self, imgs):
        """Defines the computation performed at every call when using gradcam
        utils."""
        assert self.with_cls_head
        return self._do_test(imgs)



@RECOGNIZERS.register_module()
class Recognizer3DZeroShot(BaseRecognizerZeroShot):
    """3D recognizer model framework."""

    def py_cpu_nms(self, dets, thresh):
        x1 = dets[:,0]
        x2 = dets[:,1]
        areas = x2-x1
        scores = dets[:,-1]
        keep = []
        index = scores.argsort()[::-1]
        while index.size >0:
            i = index[0]       # every time the first is the biggst, and add it directly
            keep.append(i)
            x11 = np.maximum(x1[i], x1[index[1:]])    # calculate the points of overlap 
            x22 = np.minimum(x2[i], x2[index[1:]])
            w = np.maximum(0, x22-x11)    # the weights of overlap
            overlaps = w
            ious = overlaps / (areas[i]+areas[index[1:]] - overlaps)
            idx = np.where(ious<=thresh)[0]
            index = index[idx+1]   # because index start from 1
        return keep

    def nms(self, proposals, iou_threshold, top_k):
        proposals = proposals[proposals[:, -1].argsort()[::-1]]
        proposals = proposals[:top_k]
        return proposals[self.py_cpu_nms(proposals, iou_threshold)]



    @auto_fp16()
    def pooling_and_permute(self, video_feature):
        x = torch.mean(video_feature, dim=[-2, -1])  # bs, c, n, h, w -> bs, c, n
        x = x.permute(0, 2, 1)  # bs, c, n -> bs, n, c
        return x

    @auto_fp16()
    def match_feature_dim_video_and_text(self, video_feature, text_feature):
        video_feature = self.convert_vision_feature(video_feature)
        text_feature = self.convert_language_feature(text_feature)
        return video_feature, text_feature

    @auto_fp16()
    def _extract_feat_nlp(self, text, device):
        """Extract features through a backbone.
        Args:
            imgs (torch.Tensor): The input images.

        Returns:
            torch.tensor: The extracted features.
        """
        x, attention_mask, x_global = self.nlp_backbone(text, device)
        text_output = dict(
            feature=x,
            attention_mask=attention_mask,
            global_feature=x_global,
        )
        return text_output


    @auto_fp16()
    def _extract_feat_video_and_nlp(self, video_output, text_output):
        # video_feature, text_feature = video_output['feature'], text_output['feature']
        # video_feature, text_feature = self.match_feature_dim_video_and_text(video_feature, text_feature)
        # text_attention_mask = text_output['attention_mask']
        # feature = self.vision_language_decoder(video_feature, text_feature, text_attention_mask)
        # output = dict(feature=feature)
        # return output

        video_feature, text_feature = video_output['feature'], text_output['global_feature']
        video_feature, text_feature = self.match_feature_dim_video_and_text(video_feature, text_feature)
        # video_feature = video_feature * text_feature + video_feature + text_feature
        feature = self.vision_language_decoder(video_feature)
        output = dict(feature=feature)
        return output


    @auto_fp16()
    def _extract_vision(self, data_batch):
        if self.use_vision_feature:
            video_feature = data_batch['raw_feature']  
            video_output = dict(feature=video_feature,)
        else:
            raise NotImplementedError()
        return video_output
    
    @auto_fp16()
    def _extract_language_for_train(self, data_batch, device):
        if self.use_nlp_feature:
            raise NotImplementedError()
        else:
            text = [video_meta['text'] for video_meta in data_batch['video_meta']]
            text_output = self._extract_feat_nlp(text, device)
        return text_output

    @auto_fp16()
    def _extract_language_for_test(self, text, device):
        if self.use_nlp_feature:
            raise NotImplementedError()
        else:
            text = [text]
            text_output = self._extract_feat_nlp(text, device)
        return text_output

    # def iou(self, x, y):
    def get_iou(self, d1, d2):
        #x.shape = []
        d1 = d1[:, None]
        d2 = d2[None]
        s1 = d1[..., 1] - d1[..., 0]
        s2 = d2[..., 1] - d2[..., 0]
        intersect_mins = torch.max(d1[..., 0], d2[..., 0])
        intersect_maxes = torch.min(d1[..., 1], d2[..., 1])
        intersect_area = torch.max(torch.tensor(0., device=d1.device), intersect_maxes-intersect_mins)
        iou = intersect_area / (s1 + s2 - intersect_area)
        return iou



    def ignore_pred_by_iou(self, y_pred, y_true, iou_th=0.5):                
        mask_gt = (y_true == 1) # [1, t, t]
        mask_pred = (y_pred > 0)
        
        for idx in range(len(y_true)):
            # 收集第idx下所有gt的bbox
            gt_idx = []
            mask_gt_idx = mask_gt[idx]
            i_gt_idx = self.meshgrid[0][mask_gt_idx].to(y_pred.device)
            j_gt_idx = self.meshgrid[1][mask_gt_idx].to(y_pred.device)
            start_time_relative = torch.max(torch.tensor(0.).to(y_pred.device), (-0.5+i_gt_idx)*self.temporal_gap)
            end_time_relative = torch.min(torch.tensor(1.).to(y_pred.device), (-0.5+j_gt_idx+1)*self.temporal_gap)
            gt_idx = torch.stack([start_time_relative, end_time_relative], dim=-1)
            # for idx_gt in range(len(i_gt_idx)):
            #     i_idx, j_idx = i_gt_idx[idx_gt], j_gt_idx[idx_gt]
            #     start_time_relative = torch.max(torch.tensor(0.), (-0.5+i_idx)*self.temporal_gap)
            #     end_time_relative = torch.min(torch.tensor(1.), (-0.5+j_idx+1)*self.temporal_gap)
            #     gt_idx.append([start_time_relative, end_time_relative])
            # gt_idx = torch.tensor(gt_idx, device=y_pred.device)

            # 收集第idx下所有pred的bbox
            pred_idx = []
            mask_pred_idx = mask_pred[idx]
            i_pred_idx = self.meshgrid[0][mask_pred_idx].to(y_pred.device)
            j_pred_idx = self.meshgrid[1][mask_pred_idx].to(y_pred.device)
            start_time_relative = torch.max(torch.tensor(0.).to(y_pred.device), (-0.5+i_pred_idx)*self.temporal_gap)
            end_time_relative = torch.min(torch.tensor(1.).to(y_pred.device), (-0.5+j_pred_idx+1)*self.temporal_gap)
            pred_idx = torch.stack([start_time_relative, end_time_relative, i_pred_idx, j_pred_idx], dim=-1)
            mask = y_true[idx, pred_idx[:, 2].long(), pred_idx[:, 3].long()] != 1
            pred_idx = pred_idx[mask]
            # for idx_pred in range(len(i_pred_idx)):
            #     i_idx, j_idx = i_pred_idx[idx_pred], j_pred_idx[idx_pred]
            #     if y_true[idx, i_idx, j_idx] == 1.:
            #         continue
            #     start_time_relative = torch.max(torch.tensor(0.).to(y_pred.device), (-0.5+i_idx)*self.temporal_gap)
            #     end_time_relative = torch.min(torch.tensor(1.).to(y_pred.device), (-0.5+j_idx+1)*self.temporal_gap)
            #     pred_idx.append([start_time_relative, end_time_relative, i_idx, j_idx])
            # pred_idx = torch.tensor(pred_idx, device=y_pred.device)

            # 计算iou
            if len(pred_idx) == 0:
                continue
            if len(gt_idx) == 0:
                continue
            iou_score = self.get_iou(pred_idx, gt_idx)
            iou_mask = torch.max(iou_score, dim=-1)[0] > iou_th
            pred_idx = pred_idx[iou_mask]

            y_pred[idx, pred_idx[:, 2].long(), pred_idx[:, 3].long()] = -1e10
            # for _, _, i_idx, j_idx in pred_idx:
            #     y_pred[idx, int(i_idx), int(j_idx)] = -1e10

        return y_pred


    def forward_train(self, data_batch, **kwargs):
        """Defines the computation performed at every call when training."""

        assert self.with_cls_head

        losses = dict()
        label = data_batch['label']

        video_output = self._extract_vision(data_batch)
        text_output = self._extract_language_for_train(data_batch, video_output['feature'].device)
        final_output = self._extract_feat_video_and_nlp(video_output, text_output)
        global_pointer_output = self.cls_head(final_output)
        # global_pointer = self.ignore_pred_by_iou(global_pointer, label, iou_th=0.5)
        loss_cls = self.cls_head.loss(global_pointer_output, label, **kwargs)

        losses.update(loss_cls)
        return losses




    def _do_test(self, imgs):
        # TODO
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""
        batches = imgs.shape[0]
        num_segs = imgs.shape[1]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])

        if self.max_testing_views is not None:
            total_views = imgs.shape[0]
            assert num_segs == total_views, (
                'max_testing_views is only compatible '
                'with batch_size == 1')
            view_ptr = 0
            feats = []
            while view_ptr < total_views:
                batch_imgs = imgs[view_ptr:view_ptr + self.max_testing_views]
                x = self.extract_feat(batch_imgs)
                if self.with_neck:
                    x, _ = self.neck(x)
                feats.append(x)
                view_ptr += self.max_testing_views
            # should consider the case that feat is a tuple
            if isinstance(feats[0], tuple):
                len_tuple = len(feats[0])
                feat = [
                    torch.cat([x[i] for x in feats]) for i in range(len_tuple)
                ]
                feat = tuple(feat)
            else:
                feat = torch.cat(feats)
        else:
            feat = self.extract_feat(imgs)
            if self.with_neck:
                feat, _ = self.neck(feat)

        if self.feature_extraction:
            # perform spatio-temporal pooling
            avg_pool = nn.AdaptiveAvgPool3d(1)
            if isinstance(feat, tuple):
                feat = [avg_pool(x) for x in feat]
                # concat them
                feat = torch.cat(feat, axis=1)
            else:
                feat = avg_pool(feat)
            # squeeze dimensions
            feat = feat.reshape((batches, num_segs, -1))
            # temporal average pooling
            feat = feat.mean(axis=1)
            return feat

        # should have cls_head if not extracting features
        assert self.with_cls_head
        cls_score = self.cls_head(feat)
        cls_score = self.average_clip(cls_score, num_segs)
        return cls_score


    def position2timeline(self, point_position, video_info, class_text):
        '''
        point_position.shape = [x, y, score] 
                                x in [0, self.temporal_dim-1] 
                                y in [0, self.temporal_dim-1] 
                            score in (0, 1)
        return proposal_list   
                list[dict]
                [
                    {
                    "score":0.99,
                    "segment":[1.5, 8.5],
                    "class_text": class_text,
                    },
                    ...
                    ...
                    {
                    "score":0.95,
                    "segment":[17.5, 28.5],
                    "class_text": class_text,
                    }
                ]
        '''
        if len(point_position) == 0:
            return []
        
        feature_extraction_interval = 16
        res_timeline = []
        for idx in range(len(point_position)):
            i_idx, j_idx, score_idx = point_position[idx]
            # start_time_relative = i_idx * (1/self.temporal_dim)
            # end_time_relative = (j_idx+1) * (1/self.temporal_dim)
            start_time_relative = max(0, (-0.5+i_idx)*self.temporal_gap)
            end_time_relative = min(1, (-0.5+j_idx+1)*self.temporal_gap)
            
            res_timeline.append([start_time_relative, end_time_relative, score_idx])
        res_timeline = np.array(res_timeline, dtype=float)

        res_timeline = soft_nms(
            res_timeline, 
            self.soft_nms_alpha, 
            self.soft_nms_low_threshold, 
            self.soft_nms_high_threshold, 
            self.post_process_top_k)
        # res_timeline = self.nms(res_timeline, 0.5, self.post_process_top_k)

        res_timeline = res_timeline[res_timeline[:, -1].argsort()[::-1]]
        video_duration = float(
            video_info['duration_frame'] // feature_extraction_interval *
            feature_extraction_interval
        ) / video_info['duration_frame'] * video_info['duration_second']

        proposal_list = []
        for idx in range(min(self.post_process_top_k, len(res_timeline))):
            proposal = {}
            proposal['score'] = float(res_timeline[idx, -1])
            proposal['segment'] = [
                max(0, res_timeline[idx, 0]) * video_duration,
                min(1, res_timeline[idx, 1]) * video_duration
            ]
            proposal['class_text'] = class_text
            proposal_list.append(proposal)
        return proposal_list


    def forward_test(self, data_batch, **kwargs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        '''
        proposal_list list[dict]: The updated proposals, e.g.
            [{'score': 0.9, 'segment': [0, 1]},
             {'score': 0.8, 'segment': [0, 2]},
            ...].
        output = [
            dict(
                video_name=video_info['video_name'],
                proposal_list=proposal_list)
        ]
        '''
        # if self.meshgrid[0].device != 'cpu':
        #     self.meshgrid[0].device = 'cpu'
        #     self.meshgrid[1].device = 'cpu'

        # # TODO

        proposal_list = []

        video_output = self._extract_vision(data_batch)
        text_list = np.unique([ann['label'] for ann in data_batch['video_meta'][0]['annotations']]).tolist()
        # text_list = self.label_text
        text_list = ['Amazing']
        for text in text_list:
            # print('text', text)
            text_output = self._extract_language_for_test(text, video_output['feature'].device)
            final_output = self._extract_feat_video_and_nlp(video_output, text_output)
            global_pointer_output = self.cls_head(final_output)
            
            # print('video_output', video_output['feature'].shape)
            # print('text_output', text_output['feature'].shape)
            # print('global_pointer', global_pointer.shape)
            # global_pointer = torch.sigmoid(global_pointer)
            # global_pointer += 5

            global_pointer_output_cls = torch.sigmoid(global_pointer_output['global_pointer_cls'][0])
            global_pointer_output_score = torch.sigmoid(global_pointer_output['global_pointer'][0])
            score_output = global_pointer_output_score * global_pointer_output_cls
            mask_output = score_output > 0.5
            i_output = self.meshgrid[0][mask_output].cpu().numpy()
            j_output = self.meshgrid[1][mask_output].cpu().numpy()
            score_output = score_output[mask_output].cpu().numpy()

            # mask_output = torch.sigmoid(global_pointer[0]) > 0.5
            # i_output = self.meshgrid[0][mask_output].cpu().numpy()
            # j_output = self.meshgrid[1][mask_output].cpu().numpy()
            # score_output = torch.sigmoid(global_pointer[0])[mask_output].cpu().numpy()

            point_position = np.array([[x, y, s] for x, y, s in zip(i_output, j_output, score_output)])
            result = self.position2timeline(point_position, data_batch['video_meta'][0], text)
            proposal_list += result
        proposal_list = sorted(proposal_list, key=lambda x: x['score'], reverse=True)
        print('proposal_list', proposal_list)

        video_info = dict(data_batch['video_meta'][0])
        output = [
            dict(
                video_name=video_info['video_name'],
                proposal_list=proposal_list)
        ]

        return output

    def forward_dummy(self, imgs, softmax=False):
        """Used for computing network FLOPs.

        See ``tools/analysis/get_flops.py``.

        Args:
            imgs (torch.Tensor): Input images.

        Returns:
            Tensor: Class score.
        """
        assert self.with_cls_head
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        x = self.extract_feat(imgs)

        if self.with_neck:
            x, _ = self.neck(x)

        outs = self.cls_head(x)
        if softmax:
            outs = nn.functional.softmax(outs)
        return (outs, )

    def forward_gradcam(self, imgs):
        """Defines the computation performed at every call when using gradcam
        utils."""
        assert self.with_cls_head
        return self._do_test(imgs)
