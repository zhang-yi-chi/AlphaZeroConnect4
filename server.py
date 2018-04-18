from __future__ import print_function
import socketserver
import json
import sys
from Model.game import ConnectFour
from agent import AlphaAgent

def _make_handler(game, agent):
    class BaseTCPHandler(socketserver.BaseRequestHandler):

        def _parse(self):
            raw_data = self.request[0].decode('ascii').strip().split(',')
            message_type, data = map(int, raw_data)
            return message_type, data

        def _play(self, action):
            game.do_action(action)

        def _my_play(self):
            action = agent.get_action(game)
            self._play(action)
            socket = self.request[1]
            print(action, self.client_address)
            client_address = list(self.client_address)
            client_address[1] += 1
            client_address = tuple(client_address)
            socket.sendto(str(action).encode('ascii'), client_address)

        def handle(self):
            """Handle message from client

            Two types of messages, has form "message_type,arg":
                New game: 0,is_first(0 or 1)
                Action: 1,action(int)
            """
            message_type, data = self._parse()
            print("Received:", message_type, data)
            if message_type == 0:
                game.initialize()
                agent.reset()
                if data == 0:
                    self._my_play()
            elif message_type == 1:
                self._play(data)
                self._my_play()

    return BaseTCPHandler


class Server(object):

    def __init__(self, ip, port):
        self._ip = ip
        self._port = port
        self._game = ConnectFour()
        self._agent = AlphaAgent(data='Model/save/best')
        Handler = _make_handler(self._game, self._agent)
        self._server = socketserver.UDPServer((ip, port), Handler)
        self._game = ConnectFour()

    def run(self):
        self._server.serve_forever()

    def close(self):
        self._server.shutdown()

if __name__ == "__main__":
    ip, port = "127.0.0.1", 5555

    if len(sys.argv) > 1:
        if len(sys.argv) == 2:
            ip = sys.argv[1]
        if len(sys.argv) == 3:
            port = int(sys.argv[2])

    server = Server(ip, port)
    server.run()
