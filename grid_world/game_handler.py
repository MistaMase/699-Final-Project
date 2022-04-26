import time
import threading
from grid_world.grid_world_pygame_engine import GridWorldPygameEngine
from grid_world.grid_world import GridWorld
from human_models.human_handler import HumanHandler


class GameHandler:
    """General games handler handles everything related to the games, including video settings (rendering),
    games syntax and semantics"""

    def __init__(self, game_description_file, control_hz=40, simulate_human=False, human_type=0):
        """
        We just need a games description file to initialize the whole games.
        """
        # the games content handler
        self.grid_world = GridWorld(game_description_file)

        self.simulate_human = simulate_human

        envs = []
        for goal_idx in range(len(self.grid_world.goals)):
            target_envs = []
            envs.append(target_envs)
            for target_idx in range(self.grid_world.goals[goal_idx].target_states.shape[0]):
                target_envs.append(GridWorld(game_description_file, goal_idx, target_idx))
        self.human_handler = HumanHandler(human_type, self.grid_world.goal_idx, envs)

        self.control_hz = control_hz

        # the engine is something that handles rendering, human interface
        self.engine = GridWorldPygameEngine(game_description_file, self.grid_world)

    def start_game(self):
        self.thread = threading.Thread(target=self.run, args=())
        self.thread.start()

    def run(self):
        while not self.engine.if_end:
            start_time = time.time()

            # first get an action from the human
            if self.simulate_human:
                human_input = self.human_handler.get_cmd(self.engine.get_current_state())
            else:
                # get an action from the human interface (games engine)
                human_input = self.engine.retrieve_human_input()

            # first map human input to robot action
            final_action = self.engine.input_to_action(human_input)

            if final_action is not None:
                self.engine.step(final_action)

            # check if the game has ended
            if self.grid_world.is_terminal(self.engine.get_current_state()):
                time.sleep(1)
                self.engine.if_end = True
                self.engine.thread.join()

            end_time = time.time()
            time.sleep(max(0., 1 / self.control_hz - (end_time - start_time)))
