from __future__ import print_function
from Model.Alpha.mcts import MCTSAlpha
from Model.Alpha.model import PolicyValueNet


class InteractiveAgent():

    def __init__(self):
        pass

    def get_action(self, game):
        valid_actions = game.get_valid_actions()
        action = input("input action [1 left - 7 right]:")
        action = int(action)
        while not isinstance(action, int) or action < 0 or action > 7 or (action - 1) not in valid_actions:
            action = input('input valid action [1 left - 7 right]:')
        return action - 1


class PolicyAgent():

    def __init__(self, model=None, data=None):
        if model is None or data is None:
            print("please provide model and data")
            exit()

        import tensorflow as tf
        self.pn = model

        self.sess = tf.InteractiveSession()
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, data)

    def get_action(self, game):
        return self.pn.get_action(game)

    def reset(self):
        self.pn.reset()


class AlphaAgent():

    def __init__(self, data=None):
        if data is None:
            exit()

        policy_value_net = PolicyValueNet()
        mcts_alpha = MCTSAlpha(policy_value_net.eval_probs,
                               policy_value_net.eval_value, c_puct=2, n_playout=100)
        self.policy = PolicyAgent(model=mcts_alpha, data=data)

    def get_action(self, game):
        return self.policy.get_action(game)

    def reset(self):
        self.policy.reset()
