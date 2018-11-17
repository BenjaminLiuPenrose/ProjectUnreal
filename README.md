# ProjectUnreal
Project for Berkeley DRL

run `python main.py`
<<<<<<< HEAD
or run `python main.py --env_type gym --env_name [your_game]`


Update @ 11/16
game:
(PongNoFrameskip-v4, tmp/Pong_log/[experiemnt type])
(BreakoutNoFrameskip-v4, tmp/Breakout_log/[experiemnt type])
(BeamRiderNoFrameskip-v4, tmp/BeamRider_log/[experiemnt type])
--
(QbertNoFrameskip-v4, tmp/Qbert_log/[experiemnt type])
(SpaceInvadersNoFrameskip-v4, tmp/SpaceInvaders_log/[experiemnt type])


experimet: (pc, vr, rp)
(False, False, False),
(True, False, False),
(False, True, False),
(False, False, True),
(True, True, True)


zx:
experiment: (True, True, True), (False, False, True) for Pong Breakout Beamrider

be:
experiment: (False, False, False), (True, False, False), for Pong Breakout Beamrider

sample cmd is in `run_all.sh`
atrai game list and ref rewards is `env_name_list.txt`

http://cs231n.stanford.edu/reports/2017/pdfs/610.pdf
other tasks mentioned before?
=======
or run `python3 main.py --env_type gym --env_name {}NoFrameskip-v4 --use_pixel_change True --use_value_replay True --use_reward_prediction True --max_time_step {}000000 --checkpoint_dir /tmp/{}_checkpoints --log_file /tmp/{}_log
tensorboard --logdir=/tmp/{}_log` where {} is name of Atari game
>>>>>>> 8c01deedc3a174afc0f2dbf07d305d4b223db5b6
