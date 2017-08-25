#!/usr/bin/env python

# Python imports.
import subprocess

# Other imports.
import abstraction_experiments
from simple_rl.agents import RMaxAgent, QLearnerAgent

# Global params.
track_options = False
agent_class = "ql" # one of 'ql' or 'rmax'


def spawn_subproc(task, samples, steps, grid_dim=11, max_options=50):
	'''
	Args:
		task (str)
		samples (int)
		steps (int)
		grid_dim (int)

	Summary:
		Spawns a child subprocess to run the experiment.
	'''
	cmd = ['./abstraction_experiments.py', \
							'-task=' + str(task), \
							'-samples=' + str(samples), \
							'-steps=' + str(steps), '-grid_dim=' + str(grid_dim), \
							'-agent=' + agent_class, \
							'-max_options=' + str(max_options)]
	if track_options:
		cmd += ['-track_options=True']

	subprocess.Popen(cmd)

def main():

	# Octo Grid
	spawn_subproc(task="octo", samples=1000, steps=200)

	# Grid
	spawn_subproc(task="grid", samples=500, steps=50,  grid_dim=10)

	# Four rooms
	spawn_subproc(task="four_room", samples=500, steps=200,  grid_dim=15)

	# Taxi
	spawn_subproc(task="taxi", samples=500, steps=2000,  grid_dim=4)

	# Pblocks grid
	spawn_subproc(task="pblocks", samples=100, steps=50000)

if __name__ == "__main__":
	main()