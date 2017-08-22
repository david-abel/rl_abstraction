import random
from simple_rl.tasks import FourRoomMDP

def _four_rooms(state_x, state_y, vi, actions, epsilon=0.0):
    if not isinstance(vi.mdp, FourRoomMDP):
        print "Abstraction Error: four_rooms SA only available for FourRoomMDP. (" + str(vi.mdp) + "given)." 
        quit()
    height, width = vi.mdp.width, vi.mdp.height

    if (state_x.x < width / 2.0) == (state_y.x < width / 2.0) \
        and (state_x.y < height / 2.0) == (state_y.y < height / 2.0):
        return True
    return False

def _random(state_x, state_y, vi, actions, epsilon=0.0):
    '''
    Args:
        state_x (State)
        state_y (State)
        vi (ValueIteration)
        actions (list)

    Returns:
        (bool): true randomly.
    '''
    cluster_prob = max(100.0 / vi.get_num_states(), 0.5)
    return random.random() > 0.3

def _v_approx_indicator(state_x, state_y, vi, actions, epsilon=0.0):
    '''
    Args:
        state_x (State)
        state_y (State)
        vi (ValueIteration)
        actions (list)

    Returns:
        (bool): true iff:
            max |V(state_x) - V(state_y)| <= epsilon
    '''
    return abs(vi.get_value(state_x) - vi.get_value(state_y)) <= epsilon

def _q_eps_approx_indicator(state_x, state_y, vi, actions, epsilon=0.0):
    '''
    Args:
        state_x (State)
        state_y (State)
        vi (ValueIteration)
        actions (list)

    Returns:
        (bool): true iff:
            max |Q(state_x,a) - Q(state_y, a)| <= epsilon
    '''
    for a in actions:
        q_x = vi.get_q_value(state_x, a)
        q_y = vi.get_q_value(state_y, a)

        if abs(q_x - q_y) > epsilon:
            return False

    return True