from collections import defaultdict
import math
import random

class MCTS:

    def __init__(self, exploration_weight=1):
        self.Q = defaultdict(int)
        self.N = defaultdict(int)
        self.children = dict()
        self.exploration_weight = exploration_weight

    def choose(self, node):
        assert not node.is_terminal()

        if node not in self.children:
            return node.find_random_child()

        def score(n):
            if self.N[n] == 0:
                return float("-inf")
            return self.Q[n] / self.N[n]

        return max(self.children[node], key=score)

    def do_rollout(self, node):
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._simulate(leaf)
        self._backpropagate(path, reward)

    def _select(self, node):
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                return path
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            node = self._uct_select(node)

    def _expand(self, node):
        if node in self.children:
            return
        self.children[node] = node.find_children()

    def _simulate(self, node):
        while True:
            if node.is_terminal():
                reward = node.reward()
                return reward
            node = node.find_random_child()

    def _backpropagate(self, path, reward):
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward

    def _uct_select(self, node):
        assert all(n in self.children for n in self.children[node])

        log_N_vertex = math.log(self.N[node])

        def uct(n):
            return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
                log_N_vertex / self.N[n]
            )

        return max(self.children[node], key=uct)


class Node:
    def __init__(self, data):
        self.data = data

    def find_children(self):
        children = []
        for i, x in enumerate(self.data):
            if x == 2:
                for y in [0, 1]:
                    new_data = self.data[:i] + (y,) + self.data[i + 1:]
                    children.append(Node(new_data))
        return children

    def find_random_child(self):
        pos = [i for i, x in enumerate(self.data) if x == 2]
        i = random.choice(pos)
        y = random.randint(0, 1)
        new_data = self.data[:i] + (y,) + self.data[i + 1:]
        return Node(new_data)

    def is_terminal(self):
        return False if 2 in self.data else True

    def reward(self):
        raise NotImplementedError('reward function is not implemented')

    def __hash__(self):
        return hash(self.data)

    def __eq__(node1, node2):
        return node1.data == node2.data