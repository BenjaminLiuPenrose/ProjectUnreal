# ProjectUnreal
Project for Berkeley DRL

run `python main.py`
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
