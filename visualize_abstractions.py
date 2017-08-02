#!/usr/bin/env python

# Python imports.
import sys
import random
import argparse
from collections import defaultdict

# Pygame setup.
try:
    import pygame
    pygame.init()
    title_font = pygame.font.SysFont("CMU Serif", 32)
    small_font = pygame.font.SysFont("CMU Serif", 22)

except ImportError:
    print "Error: pygame not installed (needed for visuals)."
    quit()

# Other imports.
from pygame.locals import *
from abstraction_experiments import *
import make_mdp
from simple_rl.tasks import FourRoomMDP
from simple_rl.utils import mdp_visualizer

colors = [[240, 163, 255], [113, 113, 198],[197, 193, 170],\
                [113, 198, 113],[85, 85, 85], [198, 113, 113],\
                [142, 56, 142], [125, 158, 192],[184, 221, 255],\
                [153, 63, 0], [142, 142, 56], [56, 142, 142]]

def visualize_state_abstr_grid(grid_mdp, state_abstr, scr_width=720, scr_height=720):
    '''
    Args:
        grid_mdp (GridWorldMDP)
        state_abstr (StateAbstraction)

    Summary:
        Visualizes the state abstraction.
    '''

    # Pygame init.  
    screen = pygame.display.set_mode((scr_width, scr_height))
    pygame.init()
    screen.fill((255, 255, 255))
    pygame.display.update()
    mdp_visualizer._draw_title_text(grid_mdp, screen)

    # Prep some dimensions to make drawing easier.
    scr_width, scr_height = screen.get_width(), screen.get_height()
    width_buffer = scr_width / 10.0
    height_buffer = 30 + (scr_height / 10.0) # Add 30 for title.
    cell_width = (scr_width - width_buffer * 2) / grid_mdp.width
    cell_height = (scr_height - height_buffer * 2) / grid_mdp.height
    goal_locs = grid_mdp.get_goal_locs()
    font_size = int(min(cell_width, cell_height) / 4.0)
    reg_font = pygame.font.SysFont("CMU Serif", font_size)
    cc_font = pygame.font.SysFont("Courier", font_size*2 + 2)

    # Setup states to compute abstr states later.
    state_dict = defaultdict(lambda : defaultdict(bool))
    for s in state_abstr.get_ground_states():
        state_dict[s.x][s.y] = s

    # Add colors until we have enough.
    while state_abstr.get_num_abstr_states() > len(colors):
        colors.append((random.randint(0,255), random.randint(0,255), random.randint(0,255)))

    # For each row:
    for i in range(grid_mdp.width):
        # For each column:
        for j in range(grid_mdp.height):

            top_left_point = width_buffer + cell_width*i, height_buffer + cell_height*j
            s = state_dict[i+1][grid_mdp.height - j]
            abs_state = state_abstr.phi(s)
            cluster_num = state_abstr.get_abs_cluster_num(abs_state)
            abstr_state_color = colors[cluster_num]
            r = pygame.draw.rect(screen, abstr_state_color, (top_left_point[0] + 5, top_left_point[1] + 5) + (cell_width-10, cell_height-10), 0)
            r = pygame.draw.rect(screen, (46, 49, 49), top_left_point + (cell_width, cell_height), 3)

            if grid_mdp.is_wall(i+1, grid_mdp.height - j):
                # Draw the walls.
                top_left_point = width_buffer + cell_width*i + 5, height_buffer + cell_height*j + 5
                r = pygame.draw.rect(screen, (255, 255, 255), top_left_point + (cell_width-10, cell_height-10), 0)
                text = reg_font.render("(wall)", True, (46, 49, 49))
                screen.blit(text, (top_left_point[0] + 10, top_left_point[1] + 20))

            if (i+1,grid_mdp.height - j) in goal_locs:
                # Draw goal.
                circle_center = int(top_left_point[0] + cell_width/2.0), int(top_left_point[1] + cell_height/2.0)
                circler_color = (154, 195, 157)
                pygame.draw.circle(screen, circler_color, circle_center, int(min(cell_width, cell_height) / 3.0))

                # Goal text.                
                text = reg_font.render("Goal", True, (46, 49, 49))
                offset = int(min(cell_width, cell_height) / 3.0)
                goal_text_point = circle_center[0] - font_size, circle_center[1] - font_size/1.5
                screen.blit(text, goal_text_point)

    pygame.display.flip()

def visualize_options_grid(grid_mdp, state_space, action_abstr, scr_width=720, scr_height=720):
    '''
    Args:
        grid_mdp (GridWorldMDP)
        state_space (list of State)
        action_abstr (ActionAbstraction)
    '''

    # Pygame init.  
    screen = pygame.display.set_mode((scr_width, scr_height))
    pygame.init()
    screen.fill((255, 255, 255))
    pygame.display.update()
    mdp_visualizer._draw_title_text(grid_mdp, screen)
    option_text_point = scr_width / 2.0 - (14*7), 18*scr_height / 20.0

    # Setup states to compute option init/term funcs.
    state_dict = defaultdict(lambda : defaultdict(None))
    for s in state_space:
        state_dict[s.x][s.y] = s

    # Draw inital option.
    option_index = 0
    opt_str = "Option " + str(option_index + 1) + " of " + str(len(action_abstr.get_actions())) # + ":" + str(next_option)
    option_text = title_font.render(opt_str, True, (46, 49, 49))
    screen.blit(option_text, option_text_point)
    next_option = action_abstr.get_actions()[option_index]
    visualize_option(screen, grid_mdp, state_dict, option=next_option)

    # Init rect and text.
    option_text = small_font.render("Init: ", True, (46, 49, 49))
    screen.blit(option_text, (40, option_text_point[1]))
    pygame.draw.rect(screen, colors[0], (90, option_text_point[1]) + (24, 24))

    # Init rect and text.
    option_text = small_font.render("Term: ", True, (46, 49, 49))
    screen.blit(option_text, (scr_width - 150, option_text_point[1]))
    pygame.draw.rect(screen, colors[1], (scr_width - 80, option_text_point[1]) + (24, 24))
    pygame.display.flip()

    # Keep updating options every space press.
    done = False
    while not done:
        # Check for key presses.
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                # Quit.
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN and event.key == K_RIGHT:
                # Toggle to the next option.
                option_index = (option_index + 1) % len(action_abstr.get_actions())
            elif event.type == KEYDOWN and event.key == K_LEFT:
                # Go to the previous option.
                option_index = (option_index - 1) % len(action_abstr.get_actions())
                if option_index < 0:
                    option_index = len(action_abstr.get_actions()) - 1

            next_option = action_abstr.get_actions()[option_index]
            visualize_option(screen, grid_mdp, state_dict, option=next_option)
            pygame.draw.rect(screen, (255, 255, 255), (130, option_text_point[1]) + (scr_width-290 , 50))
            opt_str = "Option " + str(option_index + 1) + " of " + str(len(action_abstr.get_actions())) # + ":" + str(next_option)
            option_text = title_font.render(opt_str, True, (46, 49, 49))
            screen.blit(option_text, option_text_point)


def visualize_option(screen, grid_mdp, state_dict, option=None):
    '''
    Args:
        screen (pygame.Surface)
        grid_mdp (GridWorldMDP)
        state_dict:
            Key: int
            Val: dict
                Key: int
                Val: state

    '''

    # Action char mapping.
    action_char_dict = {
            "up":u"\u2191",
            "down":u"\u2193",
            "left":u"\u2190",
            "right":u"\u2192"
        }

    # Prep some dimensions/fonts to make drawing easier.
    scr_width, scr_height = screen.get_width(), screen.get_height()
    width_buffer = scr_width / 10.0
    height_buffer = 30 + (scr_height / 10.0) # Add 30 for title.
    cell_width = (scr_width - width_buffer * 2) / grid_mdp.width
    cell_height = (scr_height - height_buffer * 2) / grid_mdp.height
    goal_locs = grid_mdp.get_goal_locs()
    font_size = int(min(cell_width, cell_height) / 4.0)
    reg_font = pygame.font.SysFont("CMU Serif", font_size)
    cc_font = pygame.font.SysFont("Courier", font_size*2 + 2)

    # For each row:
    for i in range(grid_mdp.width):
        # For each column:
        for j in range(grid_mdp.height):

            # Default square per state.
            top_left_point = width_buffer + cell_width*i, height_buffer + cell_height*j
            r = pygame.draw.rect(screen, (46, 49, 49), top_left_point + (cell_width, cell_height), 3)

            if grid_mdp.is_wall(i+1, grid_mdp.height - j):
                # Draw the walls.
                top_left_point = width_buffer + cell_width*i + 5, height_buffer + cell_height*j + 5
                r = pygame.draw.rect(screen, (94, 99, 99), top_left_point + (cell_width-10, cell_height-10), 0)

            if (i+1,grid_mdp.height - j) in goal_locs:
                # Draw goal.
                circle_center = int(top_left_point[0] + cell_width/2.0), int(top_left_point[1] + cell_height/2.0)
                circler_color = (154, 195, 157)
                pygame.draw.circle(screen, circler_color, circle_center, int(min(cell_width, cell_height) / 3.0))

                # Goal text.                
                text = reg_font.render("Goal", True, (46, 49, 49))
                offset = int(min(cell_width, cell_height) / 3.0)
                goal_text_point = circle_center[0] - font_size, circle_center[1] - font_size/1.5
                screen.blit(text, goal_text_point)

            if option is not None:
                # Visualize option.

                if i+1 not in state_dict.keys() or grid_mdp.height - j not in state_dict[i+1].keys():
                    # At a wall, don't draw Option stuff.
                    continue

                s = state_dict[i+1][grid_mdp.height - j]

                if option.is_init_true(s):
                    # Init.
                    r = pygame.draw.rect(screen, colors[0], (top_left_point[0] + 5, top_left_point[1] + 5) + (cell_width - 10, cell_height - 10), 0)
                elif option.is_term_true(s):
                    # Term.
                    r = pygame.draw.rect(screen, colors[1], (top_left_point[0] + 5, top_left_point[1] + 5) + (cell_width - 10, cell_height - 10), 0)
                else:
                    # White out old option.
                    r = pygame.draw.rect(screen, (255, 255, 255), (top_left_point[0] + 5, top_left_point[1] + 5) + (cell_width - 10, cell_height - 10), 0)

                # Draw option policy.
                a = option.policy(s)
                if a not in action_char_dict:
                    text_a = a
                else:
                    text_a = action_char_dict[a]
                text_center_point = int(top_left_point[0] + cell_width/2.0 - 10), int(top_left_point[1] + cell_height/4.0)
                text_rendered_a = cc_font.render(text_a, True, (46, 49, 49))
                screen.blit(text_rendered_a, text_center_point)

    pygame.display.flip()

def parse_args():
    '''
    Summary:
        Parse all arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("-abs", type = str, default="aa", nargs = '?', help = "Choose the abstraction type (one of {sa, aa}).")
    args = parser.parse_args()

    abs_type = args.abs

    return abs_type

def main():

    # MDP Setting.
    mdp_class = "four_room"
    is_sa = parse_args()

    # Single Task.
    mdp = make_mdp.make_mdp(mdp_class=mdp_class)
    actions = mdp.actions
    gamma = mdp.gamma

    # Grab SA and AA for each abstraction agent.   
    directed_sa, directed_aa = get_abstractions(mdp, directed=True)
    regular_sa, regular_aa = directed_sa, get_aa(mdp, actions, default=True)

    abs_type = parse_args()

    if abs_type == "sa":
        # Visualize State Abstractions.
        visualize_state_abstr_grid(mdp, directed_sa)
    elif abs_type == "aa":
        # Visualize Action Abstractions.
        visualize_options_grid(mdp, regular_sa.get_ground_states(), directed_aa)
    else:
        print "Error: abs type not recognized (" + abs_type + "). Options include {aa, sa}."
        quit()

    raw_input("Press any key to quit ")
    quit()

if __name__ == "__main__":
    main()
