import torch
from torch import nn

from ..builder import RECOGNIZERS
from .base import BaseRecognizer, BaseRecognizerZeroShot
from mmcv.runner import auto_fp16
import numpy as np
from ..localizers.utils import post_processing

from rich import print


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

    @auto_fp16()
    def pooling_and_permute(self, video_feature):
        x = torch.mean(video_feature, dim=[-2, -1])  # bs, c, n, h, w -> bs, c, n
        x = x.permute(0, 2, 1)  # bs, c, n -> bs, n, c
        return x

    @auto_fp16()
    def match_feature_dim_video_and_text(self, video_feature, text_feature):
        video_feature = self.convert_feature(video_feature)
        return video_feature, text_feature

    @auto_fp16()
    def _extract_feat_nlp(self, text, device):
        """Extract features through a backbone.
        Args:
            imgs (torch.Tensor): The input images.

        Returns:
            torch.tensor: The extracted features.
        """
        x, attention_mask = self.nlp_backbone(text, device)
        text_output = dict(
            feature=x,
            attention_mask=attention_mask
        )
        return text_output


    @auto_fp16()
    def _extract_feat_video_and_nlp(self, video_output, text_output):
        video_feature, text_feature = video_output['feature'], text_output['feature']
        video_feature, text_feature = self.match_feature_dim_video_and_text(video_feature, text_feature)
        text_attention_mask = text_output['attention_mask']
        feature = self.vision_language_decoder(video_feature, text_feature, text_attention_mask)
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


    def forward_train(self, data_batch, **kwargs):
        """Defines the computation performed at every call when training."""

        assert self.with_cls_head

        losses = dict()
        label = data_batch['label']

        video_output = self._extract_vision(data_batch)
        text_output = self._extract_language_for_train(data_batch, video_output['feature'].device)
        final_output = self._extract_feat_video_and_nlp(video_output, text_output)
        global_pointer = self.cls_head(final_output)
        loss_cls = self.cls_head.loss(global_pointer, label, **kwargs)

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


    def position2timeline(self, point_position, video_info):
        '''
        point_position.shape = [x, y, score] 
                                x in [0, self.temporal_dim-1] 
                                y in [0, self.temporal_dim-1] 
                            score in (0, 1)
        '''

        cur_duration = 0
        pass


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

        # # TODO

        proposal_list = []

        video_output = self._extract_vision(data_batch)
        text_list = np.unique([ann['label'] for ann in data_batch['video_meta'][0]['annotations']]).tolist()
        for text in text_list:
            text_output = self._extract_language_for_test(text, video_output['feature'].device)
            final_output = self._extract_feat_video_and_nlp(video_output, text_output)
            global_pointer = self.cls_head(final_output)
            print('video_output', video_output['feature'].shape)
            print('text_output', text_output['feature'].shape)
            print('global_pointer', global_pointer.shape)

            mask_output = global_pointer[0]>0 # [1, t, t]
            x_output = self.meshgrid[0][mask_output].cpu().numpy()
            y_output = self.meshgrid[1][mask_output].cpu().numpy()
            point_position = np.array([[x, y, 0.99999] for x, y in zip(x_output, y_output)])
            print(point_position)
            result = self.position2timeline(point_position, self.temporal_dim, data_batch['video_meta'])
            print(result)
            # proposal = self.post_processing(
            #     result,
            #     data_batch['video_meta'],
            #     self.soft_nms_alpha,
            #     self.soft_nms_low_threshold,
            #     self.soft_nms_high_threshold,
            #     self.post_process_top_k,
            # )

        proposal_list = [
            {'score': 0.9, 'segment': [0, 1]},
            {'score': 0.8, 'segment': [0, 2]},
        ]
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
