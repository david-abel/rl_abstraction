# Non-standard imports.
from simple_rl.agents.AgentClass import Agent
from StateAbstractionClass import StateAbstraction

class AbstractionWrapper(Agent):

    def __init__(self, agent, state_abs=StateAbstraction()):
        '''
        Args:
            agent (Agent)
            state_abs (func: State --> State)
        '''
        Agent.__init__(self, name=agent.name + "-sa", actions=agent.actions, gamma=agent.gamma)
        self.agent = agent
        self.state_abs = state_abs

    def act(self, ground_state, reward):
        '''
        Args:
            ground_state (State)
            reward (float)

        Return:
            (str)
        '''
        abstr_state = self.state_abs.phi(ground_state)

        action = self.agent.act(abstr_state, reward)

        return action
