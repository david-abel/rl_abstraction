'''
make_mdp.py

Utility for making MDP instances or distributions.
'''

# Python imports.
import itertools
import random
from collections import defaultdict

# Other imports.
from simple_rl.tasks import ChainMDP, GridWorldMDP, TaxiOOMDP, RandomMDP, FourRoomMDP, HanoiMDP
from simple_rl.tasks.grid_world.GridWorldMDPClass import make_grid_world_from_file
from simple_rl.mdp import MDPDistribution
from ColorMDPClass import ColorMDP

def make_mdp(mdp_class="grid", grid_dim=7):
    '''
    Returns:
        (MDP)
    '''
    # Grid/Hallway stuff.
    width, height = grid_dim, grid_dim
    upworld_goal_locs = [(i, width) for i in range(1, height+1)]

    four_room_goal_locs = [(width, height)] #, (width, 1), (1, height)] # (1, height - 2), (width - 2, height - 2), (width - 1, height - 1), (width - 2, 1)]
    four_room_goal_loc = four_room_goal_locs[0]

    # Taxi stuff.
    agent = {"x":1, "y":1, "has_passenger":0}
    passengers = [{"x":grid_dim / 2, "y":grid_dim / 2, "dest_x":grid_dim-2, "dest_y":2, "in_taxi":0}]
    walls = []

    # Do grids separately to avoid making error-prone domains.
    if mdp_class == "four_room":
        mdp = FourRoomMDP(width=width, height=height, goal_locs=[four_room_goal_loc])
    else:
        mdp = {"upworld":GridWorldMDP(width=width, height=height, init_loc=(1, 1), goal_locs=upworld_goal_locs),
            "chain":ChainMDP(num_states=grid_dim),
            "random":RandomMDP(num_states=50, num_rand_trans=2),
            "hanoi":HanoiMDP(num_pegs=grid_dim, num_discs=3),
            "taxi":TaxiOOMDP(width=grid_dim, height=grid_dim, agent=agent, walls=walls, passengers=passengers)}[mdp_class]

    return mdp

def make_mdp_distr(mdp_class="grid", grid_dim=9, horizon=0, step_cost=0, gamma=0.99):
    '''
    Args:
        mdp_class (str): one of {"grid", "random"}
        horizon (int)
        step_cost (float)
        gamma (float)

    Returns:
        (MDPDistribution)
    '''
    mdp_dist_dict = {}
    height, width = grid_dim, grid_dim

    # Define goal locations.
        
    # Corridor.
    corr_width = 20
    corr_goal_magnitude = 1 #random.randint(1, 5)
    corr_goal_cols = [i for i in xrange(1, corr_goal_magnitude + 1)] + [j for j in xrange(corr_width-corr_goal_magnitude + 1, corr_width + 1)]
    corr_goal_locs  = list(itertools.product(corr_goal_cols, [1]))

    # Grid World
    tl_grid_world_rows, tl_grid_world_cols = [i for i in xrange(width - 4, width)], [j for j in xrange(height - 4, height)]
    tl_grid_goal_locs = list(itertools.product(tl_grid_world_rows, tl_grid_world_cols))
    tr_grid_world_rows, tr_grid_world_cols = [i for i in xrange(1, 4)], [j for j in xrange(height - 4, height)]
    tr_grid_goal_locs = list(itertools.product(tr_grid_world_rows, tr_grid_world_cols))
    grid_goal_locs = tl_grid_goal_locs + tr_grid_goal_locs

    # Hallway.
    upworld_goal_locs = [(i, height) for i in xrange(1, 30)]

    # Four room.
    four_room_goal_locs = [(width, height),(width, 1), (1, height)] #, (width - width / 2 + 1, height), (width, height - height / 2 + 1), (width, 1), (width - width / 2 + 1, 1)]
                            # (width, 1 + width / 2 - 1),(1, height),(1, height - height / 2 + 1),\
                            # (1 + height / 2 - 1, height)]
                            
    tight_four_room_goal_locs = [(width, height), (width, height-1), (width-1, height), (width, height - 2), (width - 2, height), (width-1, height-1)]

    # Taxi.
    agent = {"x":1, "y":1, "has_passenger":0}
    walls = []

    goal_loc_dict = {"four_room":four_room_goal_locs,
                    "color":four_room_goal_locs,
                    "upworld":upworld_goal_locs,
                    "grid":grid_goal_locs,
                    "corridor":corr_goal_locs,
                    "tight_four_room":tight_four_room_goal_locs,
                    }

    # MDP Probability.
    num_mdps = 10 if mdp_class not in goal_loc_dict.keys() else len(goal_loc_dict[mdp_class])
    if mdp_class == "octo":
        num_mdps = 12
    mdp_prob = 1.0 / num_mdps

    for i in xrange(num_mdps):

        new_mdp = {"hrooms":make_grid_world_from_file("utils/hierarch_rooms.txt", num_goals=7, randomize=False),
                    "octo":make_grid_world_from_file("utils/octogrid.txt", num_goals=12, randomize=False, goal_num=i),
                    "upworld":GridWorldMDP(width=30, height=height, rand_init=False, goal_locs=goal_loc_dict["upworld"], name="upworld", is_goal_terminal=True),
                    "corridor":GridWorldMDP(width=20, height=1, init_loc=(10, 1), goal_locs=[goal_loc_dict["corridor"][i % len(goal_loc_dict["corridor"])]], is_goal_terminal=True, name="corridor"),
                    "grid":GridWorldMDP(width=width, height=height, rand_init=True, goal_locs=[goal_loc_dict["grid"][i % len(goal_loc_dict["grid"])]], is_goal_terminal=True),
                    "four_room":FourRoomMDP(width=width, height=height, goal_locs=[goal_loc_dict["four_room"][i % len(goal_loc_dict["four_room"])]], is_goal_terminal=True),
                    "color":ColorMDP(width=width, height=height, num_colors=4, goal_locs=[goal_loc_dict["four_room"][i % len(goal_loc_dict["four_room"])]], is_goal_terminal=True),
                    "tight_four_room":FourRoomMDP(width=width, height=height, goal_locs=[goal_loc_dict["tight_four_room"][i % len(goal_loc_dict["tight_four_room"])]], is_goal_terminal=True, name="tight_four_room")}[mdp_class]

        new_mdp.set_step_cost(step_cost)
        new_mdp.set_gamma(gamma)
        
        mdp_dist_dict[new_mdp] = mdp_prob

    return MDPDistribution(mdp_dist_dict, horizon=horizon)
