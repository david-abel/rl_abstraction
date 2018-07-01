#!/usr/bin/env python

# Python imports.
import subprocess

# Other imports.
from simple_rl.agents import QLearningAgent, DelayedQAgent

def spawn_subproc(task, samples, steps, episodes, grid_dim, agent_class):
	'''
	Args:
		task (str)
		samples (int)
		steps (int)
		steps (int)
		grid_dim (int)

	Summary:
		Spawns a child subprocess to run the experiment.
	'''
	cmd = ['./simple_sa_experiments.py', \
							'-task=' + str(task), \
							'-samples=' + str(samples), \
							'-episodes=' + str(episodes),
							'-steps=' + str(steps), \
							'-grid_dim=' + str(grid_dim), \
							'-agent=' + str(agent_class)]

	subprocess.Popen(cmd)

def main():

	episodes = 100
	samples = 100

	# Color.
		# Figure 3a
	spawn_subproc(task="color", samples=samples, episodes=episodes, steps=250,  grid_dim=11, agent_class=DelayedQAgent)
		# Figure 3b
	# spawn_subproc(task="color", samples=samples, episodes=episodes, steps=250,  grid_dim=11, agent_class=QLearningAgent)

	# Four rooms.
	# spawn_subproc(task="four_room", samples=25, episodes=100, steps=500,  grid_dim=30, agent_class=DelayedQAgent)


if __name__ == "__main__":
	main()