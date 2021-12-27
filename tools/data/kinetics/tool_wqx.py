import os
from acvp.generel_utils.tool_other import *
import glob
from rich import print
from tqdm import tqdm


def del_suffix_in_dir(data_dir):
    all_class_folder = sorted(list(glob.glob(f"{data_dir}/*")))
    for class_folder in tqdm(all_class_folder):
        all_sample_folder = sorted(list(glob.glob(f"{class_folder}/*")))
        for sample_folder in all_sample_folder:
            print(sample_folder)
            sample_folder_new = sample_folder.split('.')[:-1]
            sample_folder_new = '.'.join(sample_folder_new)
            l = f"mv {sample_folder} {sample_folder_new}"
            run_in_cmd(l)


if __name__ == '__main__':
    # import sys
    # dataset_dir = sys.argv[1]
    dataset_dir = '/new_share/wangqixun/workspace/githup_project/Video-Swin-Transformer/data/kinetics400/rawframes_val/frames/'
    del_suffix_in_dir(dataset_dir)



