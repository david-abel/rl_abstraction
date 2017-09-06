# Python imports.
from collections import defaultdict
import copy
import os

# Other imports.
from simple_rl.agents import Agent, RMaxAgent
from state_abs.StateAbstractionClass import StateAbstraction
from action_abs.ActionAbstractionClass import ActionAbstraction

class AbstractionWrapper(Agent):

    def __init__(self,
                    SubAgentClass,
                    actions,
                    mdp_name,
                    max_option_steps=0,
                    state_abstr=None,
                    action_abstr=None,
                    name_ext="abstr"):
        '''
        Args:
            SubAgentClass (simple_rl.AgentClass)
            actions (list of str)
            mdp_name (str)
            state_abstr (StateAbstraction)
            state_abstr (ActionAbstraction)
            name_ext (str)
        '''

        # Setup the abstracted agent.
        self._create_default_abstractions(actions, state_abstr, action_abstr)
        self.agent = SubAgentClass(actions=self.action_abstr.get_actions())
        self.exp_directory = os.path.join(os.getcwdu(), "results", mdp_name, "options")
        self.reward_since_tracking = 0
        self.max_option_steps = max_option_steps
        self.num_option_steps = 0
        Agent.__init__(self, name=self.agent.name + "-" + name_ext, actions=self.action_abstr.get_actions())
        self._setup_files()

    def _setup_files(self):
        '''
        Summary:
            Creates and removes relevant directories/files.
        '''
        if not os.path.exists(os.path.join(self.exp_directory)):
            os.makedirs(self.exp_directory)

        if os.path.exists(os.path.join(self.exp_directory, str(self.name)) + ".csv"):
            # Remove old
            os.remove(os.path.join(self.exp_directory, str(self.name)) + ".csv")


    def write_datum_to_file(self, datum):
        '''
        Summary:
            Writes datum to file.
        '''
        out_file = open(os.path.join(self.exp_directory, str(self.name)) + ".csv", "a+")
        out_file.write(str(datum) + ",")
        out_file.close()

    def _record_experience(self, ground_state, reward):
        '''
        Args:
            abstr_state
            abstr_action
            reward
            next_abstr_state

        Summary:
            Tracks experiences to display plots in terms of options.
        '''
        # if not self.action_abstr.is_next_step_continuing_option(ground_state):
        self.write_datum_to_file(self.reward_since_tracking)
        self.reward_since_tracking = 0

    def _create_default_abstractions(self, actions, state_abstr, action_abstr):
        '''
        Summary:
            We here create the default abstractions.
        '''
        if action_abstr is None:
            self.action_abstr = ActionAbstraction(options=agent.actions, prim_actions=agent.actions)
        else:
            self.action_abstr = action_abstr

        self.state_abstr = StateAbstraction() if state_abstr is None else state_abstr

    def act(self, ground_state, reward):
        '''
        Args:
            ground_state (State)
            reward (float)

        Return:
            (str)
        '''
        self.reward_since_tracking += reward

        if self.max_option_steps > 0:
            # We're counting action steps in terms of options.
            if self.num_option_steps == self.max_option_steps:
                # We're at the limit.
                self._record_experience(ground_state, reward)
                self.num_option_steps += 1
                return "terminate"
            elif self.num_option_steps > self.max_option_steps:
                # Skip.
                return "terminate"
            elif not self.action_abstr.is_next_step_continuing_option(ground_state):
                # Taking a new option, count it and continue.
                self.num_option_steps += 1
                self._record_experience(ground_state, reward)
        else:
            self._record_experience(ground_state, reward)

        abstr_state = self.state_abstr.phi(ground_state)
        
        # print ground_state, abstr_state, hash(ground_state)

        ground_action = self.action_abstr.act(self.agent, abstr_state, ground_state, reward)

        # print "ground_action", ground_action, type(ground_action), len(ground_action)

        return ground_action

    def reset(self):
        # Write data.
        out_file = open(os.path.join(self.exp_directory, str(self.name)) + ".csv", "a+")
        out_file.write("\n")
        out_file.close()
        self.agent.reset()
        self.action_abstr.reset()
        self.reward_since_tracking = 0
        self.num_option_steps = 0

    def new_task(self):
        self._reset_reward()

    def get_num_known_sa(self):
        return self.agent.get_num_known_sa()

    def _reset_reward(self):
        if isinstance(self.agent, RMaxAgent):
            self.agent._reset_reward()

    def end_of_episode(self):
        self.agent.end_of_episode()
        self.action_abstr.end_of_episode()
