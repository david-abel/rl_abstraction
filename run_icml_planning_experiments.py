#!/usr/bin/env python

# Python imports.
import subprocess

# Other imports.

def spawn_subproc():
	'''
	Args:

	Summary:
		Spawns a child subprocess to run the experiment.
	'''
	cmd = ['./simple_planning_experiments.py']

	subprocess.Popen(cmd)

def main():
	spawn_subproc()


if __name__ == "__main__":
	main()