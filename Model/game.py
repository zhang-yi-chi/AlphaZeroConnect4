from __future__ import print_function
from time import sleep
import numpy as np
from copy import deepcopy

# current_player 0, black, 1 in board
# current_player 1, white, 2 in board

class ConnectFour():
	def __init__(self, **kwargs):
		self.actions = list(range(7))
		if len(kwargs) == 0:
			self.initialize()
		else:
			self._initialize(board=kwargs['board'], is_black=kwargs['is_black'])
	
	def initialize(self):
		self._board = [[0 for i in range(7)] for j in range(6)]
		self._valid_actions = list(range(7))
		self._winner = None
		self._current_player = 0
		self._is_terminated = False
		# self._historical_state = np.zeros_like(self._board)

	def _initialize(self, board, is_black):
		self._board = deepcopy(board)
		self._valid_actions = [i for i in range(7) if board[0][i] == 0]
		self._winner = None
		self._current_player = 0 if is_black else 1
		self._is_terminated = False

	def get_board(self):
		return deepcopy(self._board)

	def get_current_player(self):
		return self._current_player

	def is_terminated(self):
		return self._is_terminated

	def get_winner(self):
		return self._winner

	def get_result(self):
		return self._is_terminated, self._winner

	def get_state(self):
		# state = np.zeros((5, 6, 7))
		state = np.zeros((2, 6, 7))
		board = np.array(self._board)
		state[0][board == self._current_player+1] = 1
		state[1][board == -self._current_player+2] = 1
		# state[2][self._historical_state == self._current_player+1] = 1
		# state[3][self._historical_state == -self._current_player+2] = 1
		# if self._current_player == 0:
		# 	state[2] = 1
		return state

	def get_valid_actions(self):
		return deepcopy(self._valid_actions)

	def do_two_play(self, agent1, agent2, verbose=False):
		self.initialize()
		agents = [agent1, agent2]
		if verbose:
			print(np.array(self._board))
			print()
		while not self._is_terminated:
			action = agents[self._current_player].get_action(self)
			pos = self.do_action(action)
			if verbose:
				print(np.array(self._board), pos)
				print()
				sleep(1)
		if verbose:
			print("winner:", self._winner)
			print("result:")
			print(np.array(self._board))
		return self._winner

	def do_action(self, action):
		if self._board[0][action] != 0:
			print(action, self._valid_actions)
			print(np.array(self._board))
			raise
		
		# update historical state
		# self._historical_state = np.copy(self._board)

		# do action
		for i in range(0, 7):
			if i == 6 or self._board[i][action] != 0:
				self._board[i-1][action] = self._current_player + 1
				break
		self._is_terminated, self._winner = self.check_terminal_state(i-1, action)
		self._current_player = 1 - self._current_player

		# remove invalid action
		if i - 1 == 0:
			self._valid_actions.remove(action)
		return i-1, action

	def check_terminal_state(self, x, y):
		if self._board[0].count(0) == 0:
			return True, None

		color = 1 if self._current_player-1 else 2

		count = 1 
		for j in range(y-1, -1, -1):
			if self._board[x][j] != color:
				break
			count += 1
		for j in range(y+1, 7):
			if self._board[x][j] != color:
				break
			count += 1
		if count >= 4:
			return True, self._current_player

		count = 1
		for i in range(x-1, -1, -1):
			if self._board[i][y] != color:
				break
			count += 1
		for i in range(x+1, 6):
			if self._board[i][y] != color:
				break
			count += 1
		if count >= 4:
			return True, self._current_player

		count = 1
		right_up_bound = min(x, 6-y)+1
		for i in range(1, right_up_bound):
			if self._board[x-i][y+i] != color:
				break
			count += 1
		left_down_bound = min(5-x, y) + 1
		for i in range(1, left_down_bound):
			if self._board[x+i][y-i] != color:
				break
			count += 1
		if count >= 4:
			return True, self._current_player

		count = 1
		right_down_bound = min(5-x, 6-y)+1
		for i in range(1, right_down_bound):
			if self._board[x+i][y+i] != color:
				break
			count += 1
		left_up_bound = min(x, y) + 1
		for i in range(1, left_up_bound):
			if self._board[x-i][y-i] != color:
				break
			count += 1
		if count >= 4:
			return True, self._current_player

		return False, None
