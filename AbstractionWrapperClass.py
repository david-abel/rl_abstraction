# Python imports.
from collections import defaultdict

# Other imports.
from simple_rl.agents.AgentClass import Agent
from StateAbstractionClass import StateAbstraction

# ------------ Indicator Functions ------------

def agent_q_estimate_equal(state_x, state_y, agent, epsilon=0.0):
    '''
    Args:
        state_x (State)
        state_y (State)
        agent (Agent)

    Returns:
        (bool): true iff:
            max |agent.Q(state_x,a) - agent.Q(state_y, a)| <= epsilon
    '''
    for a in agent.actions:
        q_x = agent.get_q_value(state_x, a)
        q_y = agent.get_q_value(state_y, a)

        if abs(q_x - q_y) > epsilon:
            return False

    return True

def agent_always_false(state_x, state_y, agent):
    return False

# ----------------------------------------------

class AbstractionWrapper(Agent):

    def __init__(self, agent, indicator_func=agent_q_estimate_equal, state_abs=StateAbstraction()):
        '''
        Args:
            agent (Agent)
            indicator_func (func: State x State x Agent --> {0,1})
            state_abs (func: State --> State)
        '''
        Agent.__init__(self, name=agent.name + "-sa", actions=agent.actions, gamma=agent.gamma)
        self.agent = agent
        self.state_abs = state_abs
        self.update_every_n_steps = 10
        self.steps_since_update = 0
        self.observed_states = set()
        self.indicator_func = indicator_func

    def act(self, ground_state, reward):
        '''
        Args:
            ground_state (State)
            reward (float)

        Return:
            (str)
        '''
        self.observed_states.add(ground_state)

        abstr_state = self.state_abs.phi(ground_state)

        action = self.agent.act(abstr_state, reward)

        # Update SA.
        self.steps_since_update += 1
        if self.steps_since_update >= self.update_every_n_steps:
            self.update_sa()
            self.steps_since_update = 0

        return action

    def update_sa(self):
        '''
        Summary:
            Must assume that @self.agent can give an accurate T, R, and Q after some number of steps.
        '''
        clusters = defaultdict(list)

        for state_x in self.observed_states:
            # Find state pairs that satisfy the condition.
            clusters[state_x] = [state_x]
            for state_y in self.observed_states:
                if state_x != state_y and self.indicator_func(state_x, state_y, self.agent):
                    clusters[state_x].append(state_y)
                    clusters[state_y].append(state_x)

        for state in clusters.keys():
            new_cluster = clusters[state]
            self.state_abs.make_cluster(new_cluster)

            # Destroy old so we don't double up.
            for s_prime in clusters[state]:
                clusters[s_prime] = []
