# Python imports.
import random
import Queue
from collections import defaultdict

# Other imports.
import make_mdp
import abstraction_experiments as ae
from simple_rl.planning.PlannerClass import Planner
from simple_rl.planning.ValueIterationClass import ValueIteration

class AbstractValueIteration(Planner):
    ''' AbstractValueIteration: Runs Value Iteration using a state and action abstraction '''

    def __init__(self, ground_mdp, state_abstr, action_abstr, sample_rate=1, delta=0.001, max_iterations=1000):
        '''
        Args:
            ground_mdp (MDP)
            state_abstr (StateAbstraction)
            action_abstr (ActionAbstraction)
        '''
        Planner.__init__(self, ground_mdp, name="abstr-" + str(ground_mdp))

        self.delta = delta
        self.max_iterations = max_iterations
        self.sample_rate = sample_rate

        self.state_abstr = state_abstr
        self.action_abstr = action_abstr
        self.value_func = defaultdict(float)
        self.reachability_done = False
        self.has_run_vi = False
        self._compute_reachable_state_space()

    def get_num_states(self):
        return len(self.states)      

    def get_states(self):
        if self.reachability_done:
            return self.states
        else:
            self._compute_reachable_state_space()
            return self.states

    def _compute_reachable_state_space(self):
        '''
        Summary:
            Starting with @self.start_state, determines all reachable states
            and stores them in self.states.
        '''
        state_queue = Queue.Queue()
        s_init = self.mdp.get_init_state()
        state_queue.put(s_init)
        self.states.append(s_init)
        ground_t = self.mdp.get_transition_func()

        while not state_queue.empty():
            ground_state = state_queue.get()
            for option in self.action_abstr.get_active_options(ground_state):
                # For each active option.
                
                # Take @sample_rate samples to estimate E[V]
                for samples in xrange(self.sample_rate):

                    next_state = option.act_until_terminal(ground_state, ground_t)

                    if next_state not in self.states:
                        self.states.append(next_state)
                        state_queue.put(next_state)

        self.reachability_done = True

    def plan(self, ground_state=None, horizon=100):
        '''
        Args:
            ground_state (State)
            horizon (int)

        Returns:
            (tuple):
                (list): List of primitive actions taken.
                (list): List of ground states.
                (list): List of abstract actions taken.
        '''

        ground_state = self.mdp.get_init_state() if ground_state is None else ground_state

        if self.has_run_vi is False:
            print "Warning: VI has not been run. Plan will be random."

        primitive_action_seq = []
        abstr_action_seq = []
        state_seq = [ground_state]
        steps = 0

        ground_t = self.transition_func

        # Until terminating condition is met.
        while (not ground_state.is_terminal()) and steps < horizon:

            # Compute best action, roll it out.
            next_option = self._get_max_q_action(ground_state)

            while not next_option.is_term_true(ground_state):
                # Keep applying option until it terminates.
                abstr_state = self.state_abstr.phi(ground_state)
                ground_action = next_option.act(ground_state)
                ground_state = ground_t(ground_state, ground_action)
                steps += 1
                primitive_action_seq.append(ground_action)

                state_seq.append(ground_state)

            abstr_action_seq.append(next_option)

        return primitive_action_seq, state_seq, abstr_action_seq

    def run_vi(self):
        '''
        Summary:
            Runs ValueIteration and fills in the self.value_func.           
        '''
        # Algorithm bookkeeping params.
        iterations = 0
        max_diff = float("inf")

        # Main loop.
        while max_diff > self.delta and iterations < self.max_iterations:
            max_diff = 0
            for s_g in self.get_states():
                if s_g.is_terminal():
                    continue

                max_q = float("-inf")
                for a in self.action_abstr.get_active_options(s_g):
                    # For each active option, compute it's q value.
                    q_s_a = self.get_q_value(s_g, a)
                    max_q = q_s_a if q_s_a > max_q else max_q

                # Check terminating condition.
                max_diff = max(abs(self.value_func[s_g] - max_q), max_diff)

                # Update value.
                self.value_func[s_g] = max_q

            iterations += 1

        value_of_init_state = self._compute_max_qval_action_pair(self.init_state)[0]
        
        self.has_run_vi = True

        return iterations, value_of_init_state
    
    def get_q_value(self, s_g, option):
        '''
        Args:
            s (State)
            a (Option): Assumed active option.

        Returns:
            (float): The Q estimate given the current value function @self.value_func.
        '''

        # Take samples and track next state counts.
        next_state_counts = defaultdict(int)
        reward_total = 0
        for samples in xrange(self.sample_rate): # Take @sample_rate samples to estimate E[V]
            next_state, reward, num_steps = self.do_rollout(option, s_g)
            next_state_counts[next_state] += 1
            reward_total += reward

        # Compute T(s' | s, option) estimate based on MLE and R(s, option).
        next_state_probs = defaultdict(float)
        avg_reward = 0
        for state in next_state_counts:
            next_state_probs[state] = float(next_state_counts[state]) / self.sample_rate
        
        avg_reward = float(reward_total) / self.sample_rate

        # Compute expected value.
        expected_future_val = 0
        for state in next_state_probs:
            expected_future_val += next_state_probs[state] * self.value_func[state]

        return avg_reward + self.gamma*expected_future_val

    def do_rollout(self, option, ground_state):
        '''
        Args:
            option (Option)
            ground_state (State)

        Returns:
            (tuple):
                (State): Next ground state.
                (float): Reward.
                (int): Number of steps taken.
        '''
        
        ground_t = self.mdp.get_transition_func()
        ground_r = self.mdp.get_reward_func()

        ground_action = option.act(ground_state)
        total_reward = ground_r(ground_state, ground_action)
        ground_state = ground_t(ground_state, ground_action)

        total_steps = 1
        while not option.is_term_true(ground_state):
            # Keep applying option until it terminates.
            ground_action = option.act(ground_state)
            total_reward += ground_r(ground_state, ground_action)
            ground_state = ground_t(ground_state, ground_action)
            total_steps += 1

        return ground_state, total_reward, total_steps

    def _compute_max_qval_action_pair(self, state):
        '''
        Args:
            state (State)

        Returns:
            (tuple) --> (float, str): where the float is the Qval, str is the action.
        '''
        # Grab random initial action in case all equal
        max_q_val = float("-inf")
        shuffled_option_list = self.action_abstr.get_active_options(state)[:]
        random.shuffle(shuffled_option_list)
        best_action = shuffled_option_list[0]

        # Find best action (action w/ current max predicted Q value)
        for option in shuffled_option_list:
            q_s_a = self.get_q_value(state, option)
            if q_s_a > max_q_val:
                max_q_val = q_s_a
                best_action = option

        return max_q_val, best_action

    def _get_max_q_action(self, state):
        '''
        Args:
            state (State)

        Returns:
            (str): denoting the action with the max q value in the given @state.
        '''
        return self._compute_max_qval_action_pair(state)[1]

def main():
    # MDP Setting.
    multi_task = False
    mdp_class = "grid"

    # Make single/multi task environment.
    environment = make_mdp.make_mdp_distr(mdp_class=mdp_class, num_mdps=3, horizon=30) if multi_task else make_mdp.make_mdp(mdp_class=mdp_class)
    actions = environment.get_actions()
    gamma = environment.get_gamma()

    directed_sa, directed_aa = ae.get_abstractions(environment, directed=True)
    default_sa, default_aa = ae.get_sa(environment, default=True), ae.get_aa(environment, default=True)

    vi = ValueIteration(environment)
    avi = AbstractValueIteration(environment, state_abstr=default_sa, action_abstr=default_aa)

    a_num_iters, a_val = avi.run_vi()
    g_num_iters, g_val = vi.run_vi()

    print "a", a_num_iters, a_val
    print "g", g_num_iters, g_val


if __name__ == "__main__":
    main()