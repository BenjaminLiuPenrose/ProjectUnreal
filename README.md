# ProjectUnreal
Project for Berkeley DRL

run `python main.py`
or run `python3 main.py --env_type gym --env_name {}NoFrameskip-v4 --use_pixel_change True --use_value_replay True --use_reward_prediction True --max_time_step {}000000 --checkpoint_dir /tmp/{}_checkpoints --log_file /tmp/{}_log
tensorboard --logdir=/tmp/{}_log` where {} is name of Atari game
