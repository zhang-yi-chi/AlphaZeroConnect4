from __future__ import print_function

import tensorflow as tf
import random
import numpy as np
from collections import deque

from Model.game import ConnectFour
from Model.self_play import SelfPlayer
from Model.Alpha.model import PolicyValueNet
from Model.Alpha.mcts import MCTS, MCTSAlpha


class AlphaGoZero():
    def __init__(self, policy_value_net, environment,
        learning_rate=1e-4, decay_rate=0.99, decay_freq=50,
        batch_size=64, replay_memory_size=5000,
        n_games=1000, n_steps=5, print_freq=100, eval_freq=50,
        c_puct=5, n_playout=100,
        reuse=False):
        self.environment = environment
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.decay_freq = decay_freq
        self.batch_size = batch_size
        self.replay_memory_size = replay_memory_size
        self.n_games = n_games
        self.n_steps = n_steps
        self.print_freq = print_freq
        self.eval_freq = eval_freq
        self.reuse = reuse

        self.replay_memory = deque([], replay_memory_size)
        self.policy_value_net = policy_value_net
        self.mcts_alpha = MCTSAlpha(policy_value_net.eval_probs, policy_value_net.eval_value, c_puct, n_playout)
        self.self_player = SelfPlayer(self.environment, self.mcts_alpha)

        self.n_actions = len(environment.actions)
        self.eval_n_playout = n_playout

    def get_feed_dict(self):
        batches = random.sample(self.replay_memory, self.batch_size)
        states, probs, vs = zip(*batches)
        return states, probs, vs

    def collect_data(self, n_plays=1):
        for i in range(n_plays):
            self.mcts_alpha.reset()
            data = self.self_player.do_self_play()
            self.replay_memory.extend(data)

    def train(self):
        estimated_log_probs = self.policy_value_net.get_log_probs()
        estimated_vs = self.policy_value_net.get_value()

        target_probs = tf.placeholder(tf.float32, [None, self.n_actions], name='probs')
        target_vs = tf.placeholder(tf.float32, [None, 1], name='vs')

        # loss = (estimated_vs - target_vs)^2 - target_probs * estimated_log_probs + regularization loss
        parameters = tf.trainable_variables()
        reg_loss = 1e-4 * tf.add_n(
            [tf.nn.l2_loss(v) for v in parameters if 'bias' not in v.name.lower()])
        # reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        value_loss = tf.losses.mean_squared_error(target_vs, estimated_vs)
        policy_loss = tf.reduce_mean(tf.reduce_sum(
            tf.multiply(target_probs, estimated_log_probs), axis=1))
        loss = value_loss - policy_loss + reg_loss
        
        learning_rate = tf.placeholder(tf.float32, shape=[])
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss)

        # Train
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()
        best_win_rate = 0.6
        best_before_after_win_rate = None
        if self.reuse:
            with open('save/best_win_rate', 'r') as rfile:
                best_win_rate = float(rfile.readline().strip())
                best_before_after_win_rate = rfile.readline().strip()
            print('reuse best model', best_win_rate, best_before_after_win_rate)
            self.saver.restore(self.sess, 'save/best')


        for i in range(self.n_games+1):
            self.collect_data()
            if len(self.replay_memory) < self.batch_size:
                continue

            for j in range(self.n_steps):
                input_states, input_probs, input_vs = self.get_feed_dict()
                feed_dict = {
                    target_probs:input_probs,
                    target_vs:input_vs,
                    self.policy_value_net.state:input_states,
                    self.policy_value_net.is_training:True,
                    learning_rate:self.learning_rate
                }
                op.run(feed_dict=feed_dict)

            if i % self.print_freq == 0:
                print(str(i/self.print_freq)+"/"+str(self.n_games/self.print_freq)+":", loss.eval(feed_dict), self.learning_rate)

            if (i + 1) % self.eval_freq == 0:
                before_wins, after_wins = self.evaluate_with_mcts()
                print("winning rate:", before_wins, after_wins)

                win_rate = (before_wins + after_wins) / 2
                if win_rate >= best_win_rate:
                    best_win_rate = win_rate
                    best_before_after_win_rate = str(before_wins)+'-'+str(after_wins)
                    self.save_model(best_win_rate, best_before_after_win_rate, i)
                    if win_rate >= 0.8:
                        self.eval_n_playout *= 2
                        best_win_rate = 0.6
                        print('upgrade evaluate mcts:', self.eval_n_playout)

            if (i+1) % self.decay_freq == 0 and self.learning_rate > 1e-5:
                self.learning_rate *= self.decay_rate

        self.sess.close()

    def save_model(self, best_win_rate, best_before_after_win_rate, i):
        with open('save/best_win_rate_%d'%i, 'w') as wfile:
            wfile.write(str(best_win_rate)+'\n')
            wfile.write(best_before_after_win_rate+'\n')
            wfile.write(str(self.eval_n_playout)+'\n')
        self.saver.save(self.sess, './save/best_%d'%i)

    def evaluate_with_mcts(self, n_plays=20):
        game = ConnectFour()
        mt = MCTS(n_playout=self.eval_n_playout)

        # 0 for policy value net version, 1 for original mcts
        before_wins, after_wins = [], []
        for i in range(n_plays):
            self.mcts_alpha.reset()
            mt.reset()
            winner = game.do_two_play(self.mcts_alpha, mt)
            before_wins.append(winner)
        for i in range(n_plays):
            self.mcts_alpha.reset()
            mt.reset()
            winner = game.do_two_play(mt, self.mcts_alpha)
            after_wins.append(winner)
        return 1.0*before_wins.count(0)/n_plays, 1.0*after_wins.count(1)/n_plays


if __name__ == '__main__':
    policy_value_net = PolicyValueNet()
    environment = ConnectFour()
    agz = AlphaGoZero(policy_value_net, environment,
        learning_rate=1e-3, decay_rate=0.1, decay_freq=1000,
        batch_size=1024, replay_memory_size=10000,
        n_games=10000, n_steps=1, print_freq=1, eval_freq=10,
        c_puct=2, n_playout=100,
        reuse=False)
    agz.train()
