
# Other imports.
from simple_rl.utils import make_mdp
from simple_rl.agents import RandomAgent, QLearnerAgent, RMaxAgent
from simple_rl.run_experiments import run_agents_multi_task
from HierarchyAgentClass import HierarchyAgent
from DynamicHierarchyAgentClass import DynamicHierarchyAgent
import action_abstr_stack_helpers as aa_stack_h
import hierarchy_helpers

def main():

    # ========================
    # === Make Environment ===
    # ========================
    mdp_class = "four_room"
    environment = make_mdp.make_mdp_distr(mdp_class=mdp_class, step_cost=0.01, grid_dim=15, gamma=1.0)
    actions = environment.get_actions()

    # ==========================
    # === Make SA, AA Stacks ===
    # ==========================
    sa_stack, aa_stack = hierarchy_helpers.make_hierarchy(environment, num_levels=2)

    # Debug.
    print "\n" + ("=" * 30) + "\n== Done making abstraction. ==\n" + ("=" * 30) + "\n"
    sa_stack.print_state_space_sizes()
    aa_stack.print_action_spaces_sizes()

    # ===================
    # === Make Agents ===
    # ===================
    # baseline_agent = QLearnerAgent(actions)
    agent_class = QLearnerAgent
    baseline_agent = agent_class(actions)
    rand_agent = RandomAgent(actions)
    l0_hierarch_agent = HierarchyAgent(agent_class, sa_stack=sa_stack, aa_stack=aa_stack, cur_level=0, name_ext="-$l_0$")
    l1_hierarch_agent = HierarchyAgent(agent_class, sa_stack=sa_stack, aa_stack=aa_stack, cur_level=1, name_ext="-$l_1$")
    # l2_hierarch_agent = HierarchyAgent(agent_class, sa_stack=sa_stack, aa_stack=aa_stack, cur_level=2, name_ext="-$l_2$")
    dynamic_hierarch_agent = DynamicHierarchyAgent(agent_class, sa_stack=sa_stack, aa_stack=aa_stack, cur_level=1, name_ext="-$d$")
    # dynamic_rmax_hierarch_agent = DynamicHierarchyAgent(RMaxAgent, sa_stack=sa_stack, aa_stack=aa_stack, cur_level=1, name_ext="-$d$")

    print "\n" + ("=" * 26)
    print "== Running experiments. =="
    print "=" * 26 + "\n"

    # ======================
    # === Run Experiment ===
    # ======================
    agents = [l1_hierarch_agent, dynamic_hierarch_agent, baseline_agent]
    run_agents_multi_task(agents, environment, task_samples=50, steps=50000, episodes=1, reset_at_terminal=True)


if __name__ == "__main__":
    main()