# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import threading

import signal
import math
import os
import time

from environment.environment import Environment
from model.model import UnrealModel
from train.trainer import Trainer
from train.rmsprop_applier import RMSPropApplier
# from options import get_options


import argparse

def get_options(option_type):
  """
  option_type: string
    'training' or 'display' or 'visualize'
  """
  parser = argparse.ArgumentParser()
  # Common
  parser.add_argument("--env_type", type= str, default="gym", choices=("gym", "maze", "lab"), help="environment type (lab or gym or maze)")
  parser.add_argument("--env_name", type=str, default="PongNoFrameskip-v4",  help="environment name")
  parser.add_argument("--use_pixel_change", type=bool, default=True, help="whether to use pixel change")
  parser.add_argument("--use_value_replay", type=bool, default=True, help="whether to use value function replay")
  parser.add_argument("--use_reward_prediction", type=bool, default=True, help="whether to use reward prediction")

  parser.add_argument("--checkpoint_dir", type=str, default="/tmp/unreal_checkpoints", help="checkpoint directory")

  # For training
  if option_type == 'training':
    parser.add_argument("--parallel_size", type=int, default=8, help="parallel thread size")
    parser.add_argument("--local_t_max", type=int, default=20, help="repeat step size")
    parser.add_argument("--rmsp_alpha", type=float, default=0.99, help="decay parameter for rmsprop")
    parser.add_argument("--rmsp_epsilon", type=float, default=0.1, help="epsilon parameter for rmsprop")

    parser.add_argument("--log_file", type=str, default="/tmp/unreal_log/unreal_log", help="log file directory")
    parser.add_argument("--initial_alpha_low", type=float, default=1e-4, help="log_uniform low limit for learning rate")
    parser.add_argument("--initial_alpha_high", type=float, default=5e-3, help="log_uniform high limit for learning rate")
    parser.add_argument("--initial_alpha_log_rate", type=float, default=0.5, help="log_uniform interpolate rate for learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor for rewards")
    parser.add_argument("--gamma_pc", type=float, default=0.9, help="discount factor for pixel control")
    parser.add_argument("--entropy_beta", type=float, default=0.001, help="entropy regularization constant")
    parser.add_argument("--pixel_change_lambda", type=float, default=0.001, help="pixel change lambda") # 0.05, 0.01 ~ 0.1 for lab, 0.0001 ~ 0.01 for gym
    parser.add_argument("--experience_history_size", type=int, default=2000, help="experience replay buffer size")
    parser.add_argument("--max_time_step", type=int, default=10 * 10**7, help="max time steps")
    parser.add_argument("--save_interval_step", type=int, default=100 * 1000, help="saving interval steps")
    parser.add_argument("--grad_norm_clip", type=float, default=40.0, help="gradient norm clipping")

  # For display
  if option_type == 'display':
    parser.add_argument("--frame_save_dir", type=str, deault="/tmp/unreal_frames", help="frame save directory")
    parser.add_argument("--recording", type=bool, default=False, help="whether to record movie")
    parser.add_argument("--frame_saving", type=bool, default=False, help="whether to save frames")

  args = parser.parse_args()
  return args


USE_GPU = True # To use GPU, set True

# get command line args
flags = get_options("training")

def log_uniform(lo, hi, rate):
  log_lo = math.log(lo)
  log_hi = math.log(hi)
  v = log_lo * (1-rate) + log_hi * rate
  return math.exp(v)


class Application(object):
  def __init__(self):
    pass

  def train_function(self, parallel_index, preparing):
    """ Train each environment. """

    trainer = self.trainers[parallel_index]
    if preparing:
      trainer.prepare()

    # set start_time
    trainer.set_start_time(self.start_time)

    while True:
      if self.stop_requested:
        break
      if self.terminate_reqested:
        trainer.stop()
        break
      if self.global_t > flags.max_time_step:
        trainer.stop()
        break
      if parallel_index == 0 and self.global_t > self.next_save_steps:
        # Save checkpoint
        self.save()

      diff_global_t = trainer.process(self.sess,
                                      self.global_t,
                                      self.summary_writer,
                                      self.summary_op,
                                      self.score_input)
      self.global_t += diff_global_t

  def run(self):
    device = "/cpu:0"
    if USE_GPU:
      device = "/gpu:0"

    initial_learning_rate = log_uniform(flags.initial_alpha_low,
                                        flags.initial_alpha_high,
                                        flags.initial_alpha_log_rate)

    self.global_t = 0

    self.stop_requested = False
    self.terminate_reqested = False

    action_size = Environment.get_action_size(flags.env_type,
                                              flags.env_name)

    self.global_network = UnrealModel(action_size,
                                      -1,
                                      flags.use_pixel_change,
                                      flags.use_value_replay,
                                      flags.use_reward_prediction,
                                      flags.pixel_change_lambda,
                                      flags.entropy_beta,
                                      device)
    self.trainers = []

    learning_rate_input = tf.placeholder("float")

    grad_applier = RMSPropApplier(learning_rate = learning_rate_input,
                                  decay = flags.rmsp_alpha,
                                  momentum = 0.0,
                                  epsilon = flags.rmsp_epsilon,
                                  clip_norm = flags.grad_norm_clip,
                                  device = device)

    for i in range(flags.parallel_size):
      trainer = Trainer(i,
                        self.global_network,
                        initial_learning_rate,
                        learning_rate_input,
                        grad_applier,
                        flags.env_type,
                        flags.env_name,
                        flags.use_pixel_change,
                        flags.use_value_replay,
                        flags.use_reward_prediction,
                        flags.pixel_change_lambda,
                        flags.entropy_beta,
                        flags.local_t_max,
                        flags.gamma,
                        flags.gamma_pc,
                        flags.experience_history_size,
                        flags.max_time_step,
                        device)
      self.trainers.append(trainer)

    # prepare session
    config = tf.ConfigProto(log_device_placement=False,
                            allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    self.sess = tf.Session(config=config)

    self.sess.run(tf.global_variables_initializer())

    # summary for tensorboard
    self.score_input = tf.placeholder(tf.int32)
    tf.summary.scalar("score", self.score_input)

    self.summary_op = tf.summary.merge_all()
    self.summary_writer = tf.summary.FileWriter(flags.log_file,
                                                self.sess.graph)

    # init or load checkpoint with saver
    self.saver = tf.train.Saver(self.global_network.get_vars())

    checkpoint = tf.train.get_checkpoint_state(flags.checkpoint_dir)
    if checkpoint and checkpoint.model_checkpoint_path:
      self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
      print("checkpoint loaded:", checkpoint.model_checkpoint_path)
      tokens = checkpoint.model_checkpoint_path.split("-")
      # set global step
      self.global_t = int(tokens[1])
      print(">>> global step set: ", self.global_t)
      # set wall time
      wall_t_fname = flags.checkpoint_dir + '/' + 'wall_t.' + str(self.global_t)
      with open(wall_t_fname, 'r') as f:
        self.wall_t = float(f.read())
        self.next_save_steps = (self.global_t + flags.save_interval_step) // flags.save_interval_step * flags.save_interval_step

    else:
      print("Could not find old checkpoint")
      # set wall time
      self.wall_t = 0.0
      self.next_save_steps = flags.save_interval_step

    # run training threads
    self.train_threads = []
    for i in range(flags.parallel_size):
      self.train_threads.append(threading.Thread(target=self.train_function, args=(i,True)))

    signal.signal(signal.SIGINT, self.signal_handler)

    # set start time
    self.start_time = time.time() - self.wall_t

    for t in self.train_threads:
      t.start()

    print('Press Ctrl+C to stop')
    signal.pause()

  def save(self):
    """ Save checkpoint.
    Called from therad-0.
    """
    self.stop_requested = True

    # Wait for all other threads to stop
    for (i, t) in enumerate(self.train_threads):
      if i != 0:
        t.join()

    # Save
    if not os.path.exists(flags.checkpoint_dir):
      os.mkdir(flags.checkpoint_dir)

    # Write wall time
    wall_t = time.time() - self.start_time
    wall_t_fname = flags.checkpoint_dir + '/' + 'wall_t.' + str(self.global_t)
    with open(wall_t_fname, 'w') as f:
      f.write(str(wall_t))

    print('Start saving.')
    self.saver.save(self.sess,
                    flags.checkpoint_dir + '/' + 'checkpoint',
                    global_step = self.global_t)
    print('End saving.')

    self.stop_requested = False
    self.next_save_steps += flags.save_interval_step

    # Restart other threads
    for i in range(flags.parallel_size):
      if i != 0:
        thread = threading.Thread(target=self.train_function, args=(i,False))
        self.train_threads[i] = thread
        thread.start()

  def signal_handler(self, signal, frame):
    print('You pressed Ctrl+C!')
    self.terminate_reqested = True

def main(argv):
  app = Application()
  app.run()

if __name__ == '__main__':
  tf.app.run()
