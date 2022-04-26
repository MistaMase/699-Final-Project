import os
import copy
import simplejson
import pygame
import numpy as np

from grid_world.human_input_listener import HumanInputListener


class GridWorldPygameEngine:
    """This class is to set up pygame rendering and define the human interface."""

    def __init__(self, game_description_file, grid_world, robot_handler):
        self.grid_world = grid_world
        pygame.init()  # init the pygame engine

        # load video settings from the description file
        with open(game_description_file, 'r') as json_f:
            game_description = simplejson.load(json_f)  # load json file
            self.screen_width = game_description['video_settings']['width']
            self.screen_height = game_description['video_settings']['height']

            # game specific
            self.num_grid = game_description['num_grid']
            self.robot_state = np.array(game_description['robot_state'])
            self.robot_img = pygame.image.load(os.path.join(os.path.dirname(__file__), game_description['robot_img']))
            self.goals = grid_world.goals
            self.ditches = grid_world.ditches
            self.obstacles = grid_world.obstacles
            self.current_s = copy.deepcopy(self.grid_world.get_init_state())

        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))  # set the resolution

        # pygame text
        text_y = 420
        line_spacing = 20
        self.font = pygame.font.Font('freesansbold.ttf', 18)
        self.target_texts, self.target_textRects = [], []
        for i, goal in enumerate(self.goals):
            self.target_texts.append(self.font.render('Goal {} : {}'.format(i, goal.name),
                                                     True, (255, 255, 255), (0, 0, 0)))
            self.target_textRects.append(self.target_texts[i].get_rect())
            text_y += line_spacing
            self.target_textRects[i].midleft = (180, text_y)

        text_y += line_spacing
        self.ditch_text = self.font.render('Ditch: red', True, (255, 255, 255), (0, 0, 0))
        self.ditch_textRect = self.ditch_text.get_rect()
        self.ditch_textRect.midleft = (180, text_y)

        text_y += line_spacing
        self.obs_text = self.font.render('Obstacle: grey', True, (255, 255, 255), (0, 0, 0))
        self.obs_textRect = self.obs_text.get_rect()
        self.obs_textRect.midleft = (180, text_y)

        # agent image
        block_size = int(self.screen_width / self.num_grid)
        self.resized_robot_img = pygame.transform.scale(self.robot_img, (block_size, block_size))

        # human interface
        self.human_input_listener = HumanInputListener()

        # robot handler
        self.robot_handler = robot_handler

        self.if_end = False

    def retrieve_human_input(self):
        return self.human_input_listener.retrieve_human_input()

    @staticmethod
    def input_to_action(human_input):
        #  it's the same for this simple grid world
        return human_input

    @staticmethod
    def pos_inside(pos, min_pos, max_pos):
        if min_pos[0] <= pos[0] <= max_pos[0] and min_pos[1] <= pos[1] <= max_pos[1]:
            return True
        else:
            return False

    def update_screen(self):
        """
        This is a function running in the update thread that keeps updating the rendered games. this
        also listens to the human input
        """
        self.screen.fill((0, 0, 0))  # first remove all content from the last iteration

        # draw the games specific content
        self.draw()

        observation_res = self.robot_handler.get_result()
        y = 520
        for string in observation_res:
            pred_text = self.font.render(string,
                                         True, (255, 255, 255), (0, 0, 0))
            pred_textRect = pred_text.get_rect()
            pred_textRect.midleft = (170, y)
            self.screen.blit(pred_text, pred_textRect)
            y += 20

        # draw the target, ditch and obstacle text
        for i in range(len(self.target_texts)):
            self.screen.blit(self.target_texts[i], self.target_textRects[i])
        self.screen.blit(self.ditch_text, self.ditch_textRect)
        self.screen.blit(self.obs_text, self.obs_textRect)

        # draw buttons
        button_left_rect = pygame.Rect(10, 470, 45, 45)
        button_up_rect = pygame.Rect(55, 425, 45, 45)
        button_right_rect = pygame.Rect(100, 470, 45, 45)
        button_down_rect = pygame.Rect(55, 515, 45, 45)
        pygame.draw.rect(self.screen, (200, 200, 200), button_left_rect, 0)
        pygame.draw.rect(self.screen, (200, 200, 200), button_up_rect, 0)
        pygame.draw.rect(self.screen, (200, 200, 200), button_right_rect, 0)
        pygame.draw.rect(self.screen, (200, 200, 200), button_down_rect, 0)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.if_end = True
                break
            if event.type == pygame.MOUSEBUTTONDOWN:
                if GridWorldPygameEngine.pos_inside(pygame.mouse.get_pos(), (10, 470), (55, 515)):
                    human_input = 3
                    self.human_input_listener.buf_human_input(human_input)
                elif GridWorldPygameEngine.pos_inside(pygame.mouse.get_pos(), (55, 425), (100, 470)):
                    human_input = 0
                    self.human_input_listener.buf_human_input(human_input)
                elif GridWorldPygameEngine.pos_inside(pygame.mouse.get_pos(), (100, 470), (145, 515)):
                    human_input = 1
                    self.human_input_listener.buf_human_input(human_input)
                elif GridWorldPygameEngine.pos_inside(pygame.mouse.get_pos(), (55, 515), (100, 560)):
                    human_input = 2
                    self.human_input_listener.buf_human_input(human_input)

        pygame.display.update()

    def draw(self):
        """
        This function is a callback function that will be called in the rendering thread of the games engine (pygame).
        draw the games specific content
        """

        block_size = int(self.screen_width / self.num_grid)
        for x in range(self.num_grid):
            for y in range(self.num_grid):
                rect = pygame.Rect(x * block_size, y * block_size,
                                   block_size, block_size)
                color = (200, 200, 200)
                width = 1
                pygame.draw.rect(self.screen, color, rect, width)
                for goal in self.goals:
                    if x == goal.goal_state[0] and y == goal.goal_state[1]:
                        color = goal.color
                        width = 0
                        pygame.draw.rect(self.screen, color, rect, width)
                for ditch in self.ditches:
                    if x == ditch[0] and y == ditch[1]:
                        pygame.draw.rect(self.screen, [200, 0, 0], rect, 0)
                for obstacle in self.obstacles:
                    if x == obstacle[0] and y == obstacle[1]:
                        pygame.draw.rect(self.screen, [180, 180, 180], rect, 0)
                if self.grid_world.feature_to_int([x, y]) == self.current_s or \
                        (isinstance(self.current_s, tuple) and self.grid_world.feature_to_int([x, y]) == self.current_s[-1]):
                    self.screen.blit(self.resized_robot_img, (x * block_size, y * block_size))

    def step(self, int_a):
        if isinstance(self.current_s, tuple):
            int_ns = self.grid_world.disc_tf[self.current_s][int_a]
        else:
            int_ns = self.grid_world.transition(self.current_s, int_a)
        self.current_s = int_ns

    def get_current_state(self):
        return self.current_s
