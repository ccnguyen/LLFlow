c0 train.py --opt confs/lowlight_zeus.yml
c2 train.py --opt confs/sonytif_zeus.yml
python train.py --opt confs/lowlight_local.yml
python test.py --opt /home/cindy/PycharmProjects/LLFlow/code/confs/LOL-pc.yml
python test_unpaired_lol.py

# ares'
#c5='CUDA_VISIBLE_DEVICES=0 python'
#c1='CUDA_VISIBLE_DEVICES=1 python'
#c4='CUDA_VISIBLE_DEVICES=2 python'
#c6='CUDA_VISIBLE_DEVICES=3 python'
#c7='CUDA_VISIBLE_DEVICES=4 python'
#c0='CUDA_VISIBLE_DEVICES=5 python'
#c2='CUDA_VISIBLE_DEVICES=6 python'
#c3='CUDA_VISIBLE_DEVICES=7 python'
#c8='CUDA_VISIBLE_DEVICES=8 python'