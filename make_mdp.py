# Python imports.
import itertools
import random as rnd

# Other imports.
from simple_rl.tasks import ChainMDP, GridWorldMDP, TaxiOOMDP, RandomMDP, FourRoomMDP

def make_markov_game(markov_game_class="grid_game"):

    return {"prison":PrisonersDilemmaMDP(),
            "rps":RockPaperScissorsMDP(),
            "grid_game":GridGameMDP()}[markov_game_class]

def make_mdp(mdp_class="grid", state_size=7):
    '''
    Returns:
        (MDP)
    '''
    # Grid/Hallway stuff.
    width, height = state_size, state_size
    hall_goal_locs = [(i,width) for i in range(1, height+1)]

    # Taxi stuff.
    agent = {"x":1, "y":1, "has_passenger":0}
    passengers = [{"x":state_size-3, "y":state_size / 2, "dest_x":state_size-2, "dest_y":2, "in_taxi":0}]
    walls = []

    mdp = {"hall":GridWorldMDP(width=width, height=height, init_loc=(1, 1), goal_locs=hall_goal_locs),
            "grid":GridWorldMDP(width=width, height=height, init_loc=(1, 1), goal_locs=[(state_size, state_size)]),
            "four_room":FourRoomMDP(width=width, height=height, goal_locs=[(width,height)]),
            "chain":ChainMDP(num_states=state_size),
            "random":RandomMDP(num_states=50, num_rand_trans=2),
            "taxi":TaxiOOMDP(width=state_size, height=state_size, slip_prob=0.0, agent=agent, walls=walls, passengers=passengers)}[mdp_class]

    return mdp

def make_mdp_distr(mdp_class="grid", num_mdps=15, gamma=0.99):
    '''
    Args:
        mdp_class (str): one of {"grid", "random"}
        num_mdps (int)

    Returns:
        (dict): {key:probability, value:mdp}
    '''
    mdp_distr = {}
    mdp_prob = 1.0 / num_mdps

    # Taxi stuff.   
    agent = {"x":1, "y":1, "has_passenger":0}
    walls = []

    height, width= 10, 10

    for i in range(num_mdps):
        new_mdp = {"grid":GridWorldMDP(width=width, height=height, init_loc=(1, 1), goal_locs=rnd.sample(zip(range(1, width+1),[height]*width), 1), is_goal_terminal=True, gamma=gamma),
                    "four_room":FourRoomMDP(width=8, height=8, goal_locs=rnd.sample([(1,8),(8,1),(8,8),(6,6),(6,1),(1,6)], 2), gamma=gamma),
                    "chain":ChainMDP(num_states=10, reset_val=rnd.choice([0, 0.01, 0.05, 0.1]), gamma=gamma),
                    "random":RandomMDP(num_states=40, num_rand_trans=rnd.randint(1,10), gamma=gamma),
                    "taxi":TaxiOOMDP(5, 5, slip_prob=0.0, agent=agent, walls=walls, \
                                    passengers=[{"x":2, "y":2, "dest_x":rnd.randint(1,3), "dest_y":rnd.randint(1,3), "in_taxi":0}], gamma=gamma)}[mdp_class]

        mdp_distr[new_mdp] = mdp_prob

    return mdp_distr
