from __future__ import division, print_function

import sys
from agent import InteractiveAgent, AlphaAgent
from Model.game import ConnectFour


def play(is_black=1):
    game = ConnectFour()
    i = InteractiveAgent()
    alpha_agent = AlphaAgent('Model/save/best')
    if is_black:
        game.do_two_play(i, alpha_agent, verbose=True)
    else:
        game.do_two_play(alpha_agent, i, verbose=True)

if __name__ == '__main__':
    is_black = 1
    if len(sys.argv) == 2:
        is_black = bool(int(sys.argv[1]))
    play(is_black)
