import numpy as np

class SelfPlayer():
	def __init__(self, game, agent):
		self.game = game
		self.agent = agent

	def _get_symmetric_data(self, data):
		sym_data = []
		for d in data:
			sym_d = [np.array([np.fliplr(s) for s in d[0]]), np.fliplr([d[1]])[0], d[2]]
			sym_data.append(sym_d)
		return sym_data

	def do_self_play(self, verbose=False):
		self.game.initialize()
		data = []
		while not self.game.is_terminated():
			cur_state = self.game.get_state()
			action, probs = self.agent.get_action_probs(self.game)
			pos = self.game.do_action(action)
			data.append([cur_state, probs, [0]])
			if verbose:
				print(np.array(self.game.get_board()), pos)
				print()
				sleep(1)
		if verbose:
			print("result:")
			print(np.array(self.game.get_board()))
		winner = self.game.get_winner()
		if winner is not None:
			for i in range(len(data)):
				data[i][2] = [1] if ((i % 2) == winner) else [-1]
		data.extend(self._get_symmetric_data(data))
		return data
