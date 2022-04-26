import os
import time
import numpy as np
from grid_world.grid_world_pygame_engine import GridWorldPygameEngine
from grid_world.grid_world import GridWorld
from human_models.human_handler import HumanHandler
from robot_models.robot_handler import RobotHandler

import ImageCapture


if __name__ == '__main__':
    '''
    This file initializes the grid world, the human, the robot and the pygame engine.
    The main loop updates the pygame window, takes in human input, applies robot action 
    and updates the state of the grid world.
    
    control_hz = Frequency of main loop. Reduce to slow down the visualization.
    direct_teleop = Set to True to provide human input manually. Set to False for running a simulated human.
    '''
    # start the control loop
    control_hz = 15
    direct_teleop = True
    true_human_goal, true_human_target = 0, 0  # this definies the terminal state for the pygame engine
    game_description_file = os.path.join(os.path.dirname(__file__), 'grid_world', 'rsc', 'game_multigoal.json')

    # create grid world with game rules
    grid_world = GridWorld(game_description_file, true_human_goal, true_human_target, deterministic=True)

    # populate all environments (In this homework we have 2 goals)
    envs = []
    for goal_idx in range(len(grid_world.goals)):
        target_envs = []
        envs.append(target_envs)
        for target_idx in range(len(grid_world.goals[goal_idx].target_states)):
            target_envs.append(GridWorld(game_description_file, goal_idx, target_idx, deterministic=True))

    # human_handler simulates human when input not provided
    human_handler = HumanHandler(grid_world.goal_idx, envs)

    # robot_handler
    robot_handler = RobotHandler(human_handler)

    # the engine is something that handles rendering, human interface
    engine = GridWorldPygameEngine(game_description_file, grid_world, robot_handler)

    # Create the image capture object
    ic = ImageCapture.ImageCapture()

    decisions = []
    while not engine.if_end:
        start_time = time.time()

        # update the pygame window
        engine.update_screen()

        # Display the webcam frame
        ic.live_image_overlay()

        # first get an action from the human (agent)
        if not direct_teleop:
            human_input = human_handler.get_cmd(engine.get_current_state())
        else:
            # get an action from the human interface (games engine)
            human_input = engine.retrieve_human_input()

        # first map human input to robot action
        human_action = engine.input_to_action(human_input)

        if human_action is not None:
            # after observing the human input, we can update the robot's belief for each goal
            robot_handler.update(human_action, engine.get_current_state())

            # then calculate q values for the updated belief and human action, to decide robot action
            final_action = robot_handler.get_cmd(human_action, engine.get_current_state())
        else:
            final_action = None

        if final_action is not None:
            # perform final action in the game
            engine.step(final_action)

            # append the actions and beliefs
            decisions.append([engine.get_current_state(), human_action, final_action,
                              robot_handler.observer.pi_g_traj[0], robot_handler.observer.pi_g_traj[1]])

        # check if the game has ended
        if grid_world.is_terminal(engine.get_current_state()):
            time.sleep(1)
            engine.if_end = True

        end_time = time.time()
        time.sleep(max(0., 1 / control_hz - (end_time - start_time)))

np.savetxt("decisions.csv", np.array(decisions), delimiter=",")
