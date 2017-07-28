# Python imports.
from collections import defaultdict
import copy

# Other imports.
from simple_rl.agents.AgentClass import Agent
from state_abs.StateAbstractionClass import StateAbstraction
from action_abs.ActionAbstractionClass import ActionAbstraction

class AbstractionWrapper(Agent):

    def __init__(self,
                    SubAgentClass,
                    actions,
                    state_abs=None,
                    action_abs=None,
                    learn=False,
                    name_ext="abstr"):
        '''
        Args:
            SubAgentClass (Class)
            actions (list of str)
            state_abs (StateAbstraction)
            state_abs (ActionAbstraction)
            learn (bool)
        '''
        # Setup the abstracted agent.
        self._create_default_abstractions(actions, state_abs, action_abs)
        self.agent = SubAgentClass(actions=self.action_abs.get_actions())
        Agent.__init__(self, name=self.agent.name + "-" + name_ext, actions=self.action_abs.get_actions())

    def _create_default_abstractions(self, actions, state_abs, action_abs):
        '''
        Summary:
            We here create the default abstractions.
        '''
        if action_abs is None:
            self.action_abs = ActionAbstraction(options=agent.actions, prim_actions=agent.actions)
        else:
            self.action_abs = action_abs

        if state_abs is None:
            self.state_abs = StateAbstraction()
        else:
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
        
        ground_action = self.action_abs.act(self.agent, abstr_state, ground_state, reward)

        return ground_action

    def reset(self):
        self.agent.reset()
        self.action_abs.reset()

    def new_task(self):
        self.agent._reset_reward()

    def get_num_known_sa(self):
        return self.agent.get_num_known_sa()

    def _reset_reward(self):
        self.agent._reset_reward()

    def end_of_episode(self):
        self.agent.end_of_episode()
        self.action_abs.end_of_episode()

    def make_abstract_mdp(self, mdp):
        '''
        Args:
            mdp (MDP)

        Returns:
            mdp (MDP): The abstracted MDP via self.phi.
        '''

        # DOESN't WORK BECAUSE OF OPTIONS. Fix.
        prim_actions = mdp.get_actions()
        ground_t, ground_r = mdp.get_transition_func(), mdp.get_reward_func()

        abstr_actions = self.action_abs.get_actions()
        abstr_init_state = self.state_abs.phi(mdp.get_init_state())
        abstr_trans_func = lambda s, a: self.phi(ground_t(s, a))
        abstr_reward_func = lambda s, a: ground_r(s, a)
        abstr_gamma = mdp.get_gamma()

        abstr_mdp = MDP(actions=abstr_actions,
                        transition_func=abstr_trans_func,
                        reward_func=abstr_reward_func,
                        init_state=abstr_init_state,
                        gamma=abstr_gamma)

        return abstr_mdp

    # --- Experimental/In progress Code ---

    def _learning_step(self):
        '''
        Summary:
            Performs a learning step for sa/aa.
        '''
        num_known = self.agent.get_num_known_sa()
        # Update SA.
        if num_known - self.prev_update_known_val >= self.update_every_n_known:
            self._update_sa()
            self.prev_update_known_val = num_known

    def _update_sa(self):
        '''
        Summary:
            Must assume that @self.agent can give an accurate T, R, and Q after some number of steps.
        '''
        clusters = defaultdict(list)

        for state_x in self.observed_states:
            # Find state pairs that satisfy the condition.
            clusters[state_x] = [state_x]
            for state_y in self.observed_states:
                if state_x != state_y and self.indicator_func(state_x, state_y, self.agent, self.state_abs.phi):
                    clusters[state_x].append(state_y)
                    clusters[state_y].append(state_x)

        for state in clusters.keys():
            new_cluster = clusters[state]
            if len(new_cluster) > 1:
                self.state_abs.make_cluster(new_cluster)

                # Destroy old so we don't double up.
                for s_prime in clusters[state]:
                    clusters[s_prime] = []