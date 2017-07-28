# Python imports.
import itertools
import random as r

# Other imports.
from simple_rl.tasks import ChainMDP, GridWorldMDP, TaxiOOMDP, RandomMDP, FourRoomMDP

def make_markov_game(markov_game_class="grid_game"):

    return {"prison":PrisonersDilemmaMDP(),
            "rps":RockPaperScissorsMDP(),
            "grid_game":GridGameMDP()}[markov_game_class]

def make_mdp(mdp_class="grid", state_size=4):
    '''
    Returns:
        (MDP)
    '''
    # Grid stuff
    goal_locs = [state_size-1, state_size-1]
    width, height = 15, state_size
    for element in itertools.product(range(1, width + 1), [height]):
        goal_locs.append(element)

    # Taxi stuff.
    agent = {"x":1, "y":1, "has_passenger":0}
    passengers = [{"x":2, "y":3, "dest_x":2, "dest_y":2, "in_taxi":0}]
    walls = []

    mdp = {"grid":GridWorldMDP(width=width, height=height, init_loc=(1, 1), goal_locs=goal_locs),
            "four_room":FourRoomMDP(width=width, height=height, goal_locs=[(width,height),(1,height),(width,1)]),
            "chain":ChainMDP(num_states=state_size),
            "random":RandomMDP(num_states=50, num_rand_trans=2),
            "taxi":TaxiOOMDP(2, 3, slip_prob=0.0, agent_loc=agent, walls=walls, passengers=passengers)}[mdp_class]

    return mdp

def make_mdp_distr(mdp_class="grid", num_mdps=15):
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

    height, width= 8, 12

    for i in range(num_mdps):
        new_mdp = {"grid":GridWorldMDP(width=width, height=height, init_loc=(1, 1), goal_locs=r.sample(zip(range(1, width+1),[height]*width),2), is_goal_terminal=True),
                    "four_room":FourRoomMDP(width=8, height=8, goal_locs=r.sample([(1,8),(8,1),(8,8),(6,6),(6,1),(1,6)], 2)),
                    "chain":ChainMDP(num_states=5, reset_val=r.choice([0, 0.1, 0.5, 1.1])),
                    "random":RandomMDP(num_states=40, num_rand_trans=r.randint(1,10)),
                    "taxi":TaxiOOMDP(5, 5, slip_prob=0.0, agent_loc=agent, walls=walls, \
                                    passengers=[{"x":2, "y":2, "dest_x":r.randint(1,3), "dest_y":r.randint(1,3), "in_taxi":0}])}[mdp_class]

        mdp_distr[new_mdp] = mdp_prob

    return mdp_distr
