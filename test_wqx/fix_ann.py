import os
from tqdm import tqdm
from rich import print



def fix_ann_txt(dir_, txt):
    videos = open(txt).readlines()
    videos = [v.strip() for v in videos]
    for v in tqdm(videos):
        v = v.split(' ')[0]
        if not os.path.exists(os.path.join(dir_, v)):
            print(os.path.join(dir_, v))


def get_ann_frame(dir_, output):
    with open(output, 'w') as f:
        all_class = sorted(os.listdir(dir_))
        label_dict = {all_class[idx]:idx for idx in range(len(all_class))}
        for idx in tqdm(range(len(all_class))):
            class_ = all_class[idx]
            cur_path = os.path.join(dir_, class_)
            label_class = label_dict[class_]
            all_sample = os.listdir(cur_path)
            for idx_sample in range(len(all_sample)):
                sample = all_sample[idx_sample]
                cur_path_1 = os.path.join(cur_path, sample)
                imgs = os.listdir(cur_path_1)
                nb_img = len(imgs)
                if nb_img == 0:
                    continue
                l = f"{cur_path_1} {len(imgs):d} {label_class}"
                f.write(l+'\n')


if __name__ == '__main__':
    # dir_ = '/new_share/caojinglong/video_classify/dataset/kinetics400/train/video'
    # txt = '/new_share/wangqixun/workspace/githup_project/Video-Swin-Transformer/data/kinetics400/k400_train.txt'
    # fix_ann_txt(dir_, txt)

    dir_ = '/new_share/caojinglong/video_classify/dataset/kinetics400/train/frames'
    output = '/new_share/wangqixun/workspace/githup_project/Video-Swin-Transformer/data/kinetics400/k400_train_frame.txt'
    get_ann_frame(dir_, output)

