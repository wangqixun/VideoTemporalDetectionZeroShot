from rich import print
import torch
from decord import VideoReader
from decord import cpu, gpu
import cv2
import numpy as np
import io

from mmcv import Config, DictAction
from mmcv.parallel import collate, scatter
from mmcv.fileio import FileClient

from mmaction.apis import inference_recognizer, init_recognizer
from mmaction.datasets.pipelines import Compose
from mmaction.datasets.pipelines.loading import SampleFrames
from mmaction.datasets.builder import PIPELINES

def get_model_mmaction2(config, checkpoint, cfg_options=None, device=None):
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        pass
    if cfg_options is None:
        cfg_options = {}
    
    cfg = Config.fromfile(config)
    cfg.merge_from_dict(cfg_options)
    model = init_recognizer(cfg, checkpoint, device=device)
    return model


def _infer_clip_video_file(video_file):
    # mmaction
    cfg = model.cfg
    start_index = cfg.data.test.get('start_index', 0)
    test_pipeline = Compose(cfg.data.test.pipeline)
    device = next(model.parameters()).device  # model device
    data = dict(
        filename=video_path,
        label=-1,
        start_index=start_index,
        modality='RGB')
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)  # batch
    if next(model.parameters()).is_cuda: # 放gpu
        data = scatter(data, [device])[0]
    with torch.no_grad():
        scores = model(return_loss=False, **data)[0]
    print(scores)


# @profile
def ttt():
    device = 'cuda:0'
    config = '/new_share/wangqixun/workspace/githup_project/Video-Swin-Transformer/configs/recognition/swin/swin_base_patch244_window877_kinetics400_22k.py'
    checkpoint = '/new_share/wangqixun/workspace/githup_project/model_super_strong/mmaction2/swin_base_patch244_window877_kinetics400_22k.pth'
    model = get_model_mmaction2(config, checkpoint)

    clip_time = 10 # 秒
    pipline_infer = model.cfg.data.test.pipeline[1:]
    pipline_infer = Compose(pipline_infer)
    # video_path = '/new_share/wangqixun/xiuchang/video_live/lives_fix/0.mp4'
    video_path = '/share/wangqixun/xiuchang/video_live/live_parts_211209/live-montage-8870313604_1639034553023_1639034612941_1639035179_8870313604_1639034536456_1639035136456_1898111708_2_1_create_type.mp4'
    file_obj = io.BytesIO(FileClient('disk').get(video_path))
    vr = VideoReader(file_obj, num_threads=16)
    # vr = VideoReader(video_path, num_threads=32)
    fps = vr.get_avg_fps()
    nb_framnes = len(vr)
    duration = nb_framnes/fps

    results_clip = {}
    for idx_vr in range(0, nb_framnes, int(fps*clip_time)):
        if (idx_vr + int(fps*clip_time)) >= nb_framnes:
            nb_framnes_clip = nb_framnes - idx_vr
        else:
            nb_framnes_clip = int(fps*clip_time)
        start_time, end_time = idx_vr/fps, (idx_vr+nb_framnes_clip)/fps
        results_clip = {
            'total_frames': nb_framnes_clip, 
            'video_reader': vr, 
            'start_index': idx_vr,
            'modality': 'RGB', 
            'label': -1
        }
        results_clip = pipline_infer(results_clip)
        results_clip['imgs'] = results_clip['imgs'].to(device)[None]
        with torch.no_grad():
            scores = model(return_loss=False, **results_clip)[0]

        scores = torch.tensor(scores)
        scores = torch.topk(scores, k=5)
        print(start_time, end_time, scores)


if __name__ == '__main__':
    ttt()



















