
# Other imports.
from simple_rl.utils import make_mdp
from simple_rl.agents import RandomAgent, QLearnerAgent
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
    environment = make_mdp.make_mdp_distr(mdp_class=mdp_class, grid_dim=11)
    actions = environment.get_actions()

    # ==========================
    # === Make SA, AA Stacks ===
    # ==========================
    # sa_stack, aa_stack = aa_stack_h.make_random_sa_diropt_aa_stack(environment, max_num_levels=3)
    sa_stack, aa_stack = hierarchy_helpers.make_hierarchy(environment, num_levels=2)

    # Debug.
    print "\n" + ("=" * 30)
    print "== Done making abstraction. =="
    print "=" * 30 + "\n"
    sa_stack.print_state_space_sizes()
    print "Num Action Abstractions:", len(aa_stack.get_aa_list())

    # ===================
    # === Make Agents ===
    # ===================
    baseline_agent = QLearnerAgent(actions)
    rand_agent = RandomAgent(actions)
    l0_hierarch_agent = HierarchyAgent(QLearnerAgent, sa_stack=sa_stack, aa_stack=aa_stack, cur_level=0, name_ext="-$l_0$")
    l1_hierarch_agent = HierarchyAgent(QLearnerAgent, sa_stack=sa_stack, aa_stack=aa_stack, cur_level=1, name_ext="-$l_1$")
    # l2_hierarch_agent = HierarchyAgent(QLearnerAgent, sa_stack=sa_stack, aa_stack=aa_stack, cur_level=2, name_ext="-$l_2$")
    dynamic_hierarch_agent = DynamicHierarchyAgent(QLearnerAgent, sa_stack=sa_stack, aa_stack=aa_stack, cur_level=1, name_ext="-$d$")

    print "\n" + ("=" * 26)
    print "== Running experiments. =="
    print "=" * 26 + "\n"

    # ======================
    # === Run Experiment ===
    # ======================
    agents = [l0_hierarch_agent, l1_hierarch_agent, dynamic_hierarch_agent, rand_agent, baseline_agent]
    run_agents_multi_task(agents, environment, task_samples=2000, steps=1000, episodes=1, reset_at_terminal=True)


if __name__ == "__main__":
    main()