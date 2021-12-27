cd ../
python demo/demo.py \
    configs/recognition/swin/swin_base_patch244_window877_kinetics400_22k.py \
    /new_share/wangqixun/workspace/githup_project/model_super_strong/mmaction2/swin_base_patch244_window877_kinetics400_22k.pth \
    demo/demo.mp4 \
    demo/label_map_k400.txt
