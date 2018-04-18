from __future__ import print_function
import numpy as np
from copy import deepcopy


class TreeNode():
	def __init__(self, parent, action, prior_p):
		self.parent = parent
		self._action = action
		self._p = prior_p

		self.children = {}
		self._q = 0.0
		self._n_visits = 0

	def is_leaf(self):
		return len(self.children) == 0

	def is_root(self):
		return self.parent is None

	def get_n_visits(self):
		return self._n_visits

	def select(self, c_puct):
		# return (action_id, action_node)
		return max(self.children.items(), key=lambda node: node[1].get_value(c_puct))

	def expand(self, action_priors):
		for action, prior_p in action_priors:
			self.children[action] = TreeNode(self, action, prior_p)

	def get_value(self, c_puct):
		u = c_puct * self._p * np.sqrt(self.parent.get_n_visits()) / (1 + self._n_visits)
		return self._q + u

	def update(self, v):
		if self.parent is not None:
			self.parent.update(-v)
		self._q = (self._n_visits * self._q + v) / (1+self._n_visits)
		self._n_visits += 1


def uniform_policy(game):
	valid_actions = game.get_valid_actions()
	probs = np.ones(len(valid_actions)) / len(valid_actions)
	return zip(valid_actions, probs)


class MCTS():
	def __init__(self, policy=uniform_policy, c_puct=5, n_playout=100):
		self._eval_probs = policy
		self._c_puct = c_puct
		self._n_playout = n_playout

		self._root = TreeNode(None, None, 1.0)

	def reset(self):
		self._root = TreeNode(None, None, 1.0)

	def _get_rollout_result(self, game):
		def rollout(game):
			valid_actions = game.get_valid_actions()
			value = np.random.rand(len(valid_actions))
			# print "rollout", value, valid_actions
			return zip(valid_actions, value)

		cur_player = game.get_current_player()
		is_terminated, winner = game.get_result()
		while not is_terminated:
			action_probs = 	rollout(game)
			action = max(action_probs, key=lambda x: x[1])[0]
			game.do_action(action)
			is_terminated, winner = game.get_result()
		res = 0
		if winner is not None:
			res = 1 if winner == cur_player else -1
		return res

	def _search(self, game):
		cur_node = self._root
		cur_player = game.get_current_player()
		while not cur_node.is_leaf():
			action, cur_node = cur_node.select(self._c_puct)
			game.do_action(action)
		is_terminated, _ = game.get_result()
		if not is_terminated:
			action_probs = self._eval_probs(game)
			cur_node.expand(action_probs)
		return cur_node

	def _playout(self, game):
		cur_node = self._search(game)
		v = self._get_rollout_result(game)
		cur_node.update(-v)

	def get_action(self, game, temp=1):
		for i in range(self._n_playout):
			cur_game = deepcopy(game)
			self._playout(cur_game)
		action_n_visits = [(action, node.get_n_visits()) for action, node in self._root.children.items()]
		action = max(action_n_visits, key=lambda x: x[1])[0]
		self.reset()
		return action


class TreeNodeAlpha(TreeNode):
	def expand(self, action_priors):
		action_priors = list(action_priors)
		noises = np.random.dirichlet(0.03 * np.ones(len(action_priors)))
		for i, (action, prior_p) in enumerate(action_priors):
			prior_p = 0.75 * prior_p + 0.25 * noises[i]
			self.children[action] = TreeNodeAlpha(self, action, prior_p)


class MCTSAlpha(MCTS):
	def __init__(self, eval_probs, eval_value, c_puct=5, n_playout=100):
		self._eval_probs = eval_probs
		self._eval_value = eval_value
		self._c_puct = c_puct
		self._n_playout = n_playout
		self.depth = 0

		self._root = TreeNodeAlpha(None, None, 1.0)

	def reset(self):
		self.depth = 0
		self._root = TreeNodeAlpha(None, None, 1.0)

	def _playout(self, game):
		cur_node = self._search(game)
		is_terminated, winner = game.get_result()
		v = 0.0
		if is_terminated:
			if winner is not None:
				v = 1.0 if winner == game.get_current_player() else -1.0
		else:
			v = self._eval_value(game)
		cur_node.update(-v)

	def _play(self, action):
		if action is not None:
			self._root = self._root.children[action]
			self._root.parent = None
		else:
			self._root = TreeNodeAlpha(None, None, 1.0)
		self.depth += 1

	def _get_action_probs(self, game, temp=1):
		def normalize_probs(x):
			x = np.power(x, 1.0 / temp)
			return x / np.sum(x)

		for i in range(self._n_playout):
			cur_game = deepcopy(game)
			self._playout(cur_game)

		action_n_visits = [(action, node.get_n_visits()) for action, node in self._root.children.items()]
		actions, n_visits = zip(*action_n_visits)
		_probs = normalize_probs(n_visits)
		action = np.random.choice(actions, p=_probs)

		valid_actions = game.get_valid_actions()
		probs = np.zeros(len(game.actions))
		probs[valid_actions] = _probs

		return action, probs

	def get_action_probs(self, game, temp=1):
		action, probs = self._get_action_probs(game, temp)
		self._play(action)
		return action, probs

	def get_action(self, game, temp=1):
		_, probs = self._get_action_probs(game, temp)
		action = np.argmax(probs)
		self._play(None)
		return action
