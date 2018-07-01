#!/usr/bin/env python

# Python imports.
import subprocess

# Other imports.
import abstraction_experiments
from simple_rl.agents import RMaxAgent, QLearningAgent

# Global params.
track_options = False
agent_class = "ql" # one of 'ql' or 'rmax'
episodic = True

def spawn_subproc(task, samples, steps, episodes=1, grid_dim=11, max_options=20):
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
	cmd = ['./abstraction_experiments.py', \
							'-task=' + str(task), \
							'-samples=' + str(samples), \
							'-episodes=' + str(episodes),
							'-steps=' + str(steps), \
							'-grid_dim=' + str(grid_dim), \
							'-agent=' + agent_class, \
							'-max_options=' + str(max_options),
							'-exp_type=combo']
	if track_options:
		cmd += ['-track_options=True']

	subprocess.Popen(cmd)

def main():

	episodes = 1
	step_multipler = 2
	if episodic:
		episodes = 100
		step_multipler = 1

	# Hall.
	# spawn_subproc(task="hall", samples=500, episodes=episodes, steps=25 * step_multipler,  grid_dim=15, max_options=32)

	# spawn_subproc(task="grid", samples=5, episodes=episodes, steps=50 * step_multipler, grid_dim=7, max_options=300)

	# Octo Grid.
	# spawn_subproc(task="octo", samples=500, episodes=episodes, steps=50 * step_multipler, max_options=50)

	spawn_subproc(task="rock_climb", samples=100, episodes=episodes, steps=50 * step_multipler, max_options=50)

	# Four rooms.
	# spawn_subproc(task="four_room", samples=300, episodes=episodes, steps=50 * step_multipler,  grid_dim=15)

	# Grid random init.
	# spawn_subproc(task="whirlpool", samples=150, episodes=episodes, steps=50 * step_multipler, grid_dim=21, max_options=200)

	# # Ice Rink.
	# spawn_subproc(task="icerink", samples=300, episodes=episodes, steps=100 * step_multipler,  grid_dim=10, max_options=40)


if __name__ == "__main__":
	main()