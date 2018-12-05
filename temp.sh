# python3 main.py --env_type gym --env_name PongNoFrameskip-v4 --use_pixel_change False --use_value_replay True --use_reward_prediction True --max_time_step 10000000 --checkpoint_dir /tmp/Pong_ftt_checkpoints --log_file /tmp/Pong_ftt_log
# # tensorboard --logdir /tmp/Pong_ftt_log

python3 main.py --env_type gym --env_name BreakoutNoFrameskip-v4 --use_pixel_change False --use_value_replay True --use_reward_prediction True --max_time_step 10000000 --checkpoint_dir /tmp/Breakout_ftt_checkpoints --log_file /tmp/Breakout_ftt_log
# tensorboard --logdir /tmp/Breakout_tff_log

# python3 main.py --env_type gym --env_name SeaquestNoFrameskip-v4 --use_pixel_change False --use_value_replay True --use_reward_prediction False --max_time_step 10000000 --checkpoint_dir /tmp/Seaquest_ftf_checkpoints --log_file /tmp/Seaquest_ftf_log
# tensorboard --logdir /tmp/Qbert_fff_log
