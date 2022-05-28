import numpy as np
from heapq import heappop, heappush
import gym
from pogema.animation import AnimationMonitor
from pogema import GridConfig

class Node:
    def __init__(self, coord: (int, int) = (0, 0), g: int = 0, h: int = 0):
        self.i, self.j = coord
        self.g = g
        self.h = h
        self.f = g + h

    def __lt__(self, other: 'Node') -> bool:
        return self.f < other.f or ((self.f == other.f) and (self.g < other.g))


grid_config = GridConfig(
    num_agents=128,
    size=64,
    density=0.4,
    seed=1,
    max_episode_steps=256,
    obs_radius=5,
)
env = gym.make("Pogema-v0", grid_config=grid_config)


class a:
    def __init__(self):
        self.start = (0, 0)
        self.goal = (0, 0)
        self.max_steps = 10000
        self.OPEN = []
        self.CLOSED = dict()
        self.obstacles = set()
        self.other_agents = set()

    def compute_shortest_path(self, start, goal):
        self.start = start
        self.goal = goal
        self.CLOSED = dict()
        self.OPEN = []
        heappush(self.OPEN, Node(self.start))
        u = Node()
        steps = 0
        while len(self.OPEN) > 0 and steps < self.max_steps and (u.i, u.j) != self.goal:
            u = heappop(self.OPEN)
            steps += 1
            for x, y in {(-1, 0), (1, 0), (0, -1), (0, 1)}:
                n = (u.i + x, u.j + y)
                if n not in self.obstacles and n not in self.CLOSED and n not in self.other_agents:

                    h = abs(n[0] - self.goal[0]) + abs(n[1] - self.goal[1])
                    heappush(self.OPEN, Node(n, u.g + 1, h))
                    self.CLOSED[n] = (u.i, u.j)

    def get_next_node(self):
        next_node = self.start
        if self.goal in self.CLOSED:
            next_node = self.goal

            while self.CLOSED[next_node] != self.start:
                next_node = self.CLOSED[next_node]
        return next_node

    def update_obstacles(self, obs, other_agents, n):
        obstacles = np.transpose(np.nonzero(obs))
        for obstacle in obstacles:
            self.obstacles.add((n[0] + obstacle[0], n[1] + obstacle[1]))
        self.other_agents.clear()
        agents = np.transpose(np.nonzero(other_agents))
        for agent in agents:
            self.other_agents.add((n[0] + agent[0], n[1] + agent[1]))
        # for agent in agents:
        #     self.other_agents.add((n[0] + agent[1], n[1] + agent[1]))
        # for agent in agents:
        #     self.other_agents.add((n[0] + agent[0], n[-1] + agent[1]))


class Model:
    def __init__(self):
        self.agents = None

        self.actions = {tuple(GridConfig().MOVES[i]): i for i in range(len(GridConfig().MOVES))}

    def act(self, obs, dones, positions_xy, targets_xy) -> list:
        if self.agents is None:
            self.agents = [a() for _ in range(len(obs))]
        actions = []
        for k, cur_obs in enumerate(obs):
            cur_agent = self.agents[k]
            cur_position = positions_xy[k]
            cur_x_position, cur_y_position = positions_xy[k]

            if cur_position == targets_xy[k]:

                actions.append(0)
                continue
            elif cur_position == positions_xy[k - 2] or cur_position == positions_xy[k - 1]:
                cur_agent.update_obstacles(obs[k - 2][1], cur_obs[1], (cur_y_position - 20, cur_y_position - 30))
                cur_agent.get_next_node()
            cur_agent.update_obstacles(cur_obs[0], cur_obs[1], (cur_x_position - 5, cur_y_position - 5))
            cur_agent.compute_shortest_path(start=cur_position, goal=targets_xy[k])
            next_node = cur_agent.get_next_node()
            actions.append(self.actions[(next_node[0] - cur_x_position, next_node[1] - cur_y_position)])
        return actions

disable_env_checker=True
def main():


    i = AnimationMonitor(env)


    obs = i.reset()

    done = [False for _ in range(len(obs))]
    solver = Model()
    steps = 0
    while not all(done):

        obs, reward, done, info = i.step(
            solver.act(
                obs=obs, dones=done, positions_xy=i.get_agents_xy_relative(), targets_xy=i.get_targets_xy_relative()
            )
        )
        steps += 1
        # print('////////////////////////////')
        # print(steps, np.sum(done))
        # print(env.get_agents_xy())



    i.save_animation("render.svg", egocentric_idx=None)


if __name__ == "__main__":
    main()