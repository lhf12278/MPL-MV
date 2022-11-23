#!/usr/bin/env bash

{
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch
python main.py   --train_dataset ['duke_market'] --s_camera_num 8 --t_camera_num 6 --pid_num 702 --steps 1033  --steps_domain 800 --base_learning_rate 0.0002 --m_2k_learn_rate 0.00012 --d_domain_learn_rate 0.0003
}
