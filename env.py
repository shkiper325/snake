import random
import numpy as np
import pygame

class Env(object):
    def __init__(self, params):
        #Initializing game parameters
        self.border_width = params['border_width']
        self.square_size = params['square_size']

        self.snake_color = np.array(params['snake_color'])
        self.food_color = np.array(params['food_color'])
        self.background_color = np.array(params['background_color'])
        self.border_background_color = np.array(params['border_background_color'])
        self.border_line_color = np.array(params['border_line_color'])
        self.snake_head_color = np.array(params['snake_head_color'])

        self.border_line_width = params['border_line_width']

        self.field_size = np.array(params['field_size'])

        self.snake_init_len = params['snake_init_len']
        self.food_count = params['food_count']

        self.food_score = params['food_score']
        self.death_score = params['death_score']
        self.survive_score = params['survive_score']

        self.torus = params['torus']

        #Initializing game state
        self.game_finished = True

        #Initializing screen
        pygame.init()

        self.screen_size = np.array([self.border_width * 2 + self.field_size[0] * self.square_size,
                self.border_width * 2 + self.field_size[1] * self.square_size])

        self.screen = pygame.display.set_mode(self.screen_size)
        
        self.screen.fill(self.background_color)
        pygame.draw.rect(self.screen, self.border_line_color, pygame.Rect(
            self.border_width - self.border_line_width,
            self.border_width - self.border_line_width,
            self.square_size * self.field_size[0] + 2 * self.border_line_width,
            self.square_size * self.field_size[1] + 2 * self.border_line_width),
            self.border_line_width
        )

    def generate_food(self):
        while len(self.food) < self.food_count:
            x = random.randint(0, self.field_size[0] - 1)
            y = random.randint(0, self.field_size[1] - 1)

            if [x, y] in self.snake or [x, y] in self.food:
                continue
            else:
                self.food.append([x, y])

    def new_game(self):
        self.snake = [[i, self.field_size[1] // 2] for i in range(self.snake_init_len)]

        self.food = []
        self.generate_food()

        self.direction = 0

        self.game_finished = False

        self.score = 0

        self.rendered = False

    def act(self, action=None):
        if self.game_finished:
            return

        if action is not None:
            if self.direction % 2 != action % 2: #Check for incompatible action
                self.direction = action

        snake_append = self.snake[-1].copy()

        if self.direction == 0:
            snake_append[0] += 1
        elif self.direction == 1:
            snake_append[1] -= 1
        elif self.direction == 2:
            snake_append[0] -= 1
        else:
            snake_append[1] += 1

        if self.torus:
            snake_append[0] %= self.field_size[0]
            snake_append[1] %= self.field_size[1]

        reward = None
        if snake_append in self.food:
            self.snake.append(snake_append)

            self.food.remove(snake_append)
            self.generate_food()

            reward = self.food_score 
        elif snake_append[0] < 0 or snake_append[0] >= self.field_size[0] or \
                snake_append[1] < 0 or snake_append[1] >= self.field_size[1]:
            self.game_finished = True

            reward = self.death_score
        elif snake_append in self.snake[1:]:
            self.game_finished = True

            reward = self.death_score
        else:
            self.snake.append(snake_append)

            self.snake.pop(0)

            reward = self.survive_score

        self.score += reward

        self.rendered = False

        return reward

    def render(self):
        if self.rendered:
            return

        pygame.draw.rect(self.screen, self.background_color, pygame.Rect(
                self.border_width, self.border_width,
                self.square_size * self.field_size[0],
                self.square_size * self.field_size[1])
        )

        for snake_point in self.snake[:-1]:
            pygame.draw.rect(self.screen, self.snake_color, pygame.Rect(
                    self.border_width + snake_point[0] * self.square_size,
                    self.border_width + snake_point[1] * self.square_size,
                    self.square_size,
                    self.square_size
            ))

        pygame.draw.rect(self.screen, self.snake_head_color, pygame.Rect(
                self.border_width + self.snake[-1][0] * self.square_size,
                self.border_width + self.snake[-1][1] * self.square_size,
                self.square_size,
                self.square_size
        ))

        for f in self.food:
            pygame.draw.rect(self.screen, self.food_color, pygame.Rect(
                    self.border_width + f[0] * self.square_size,
                    self.border_width + f[1] * self.square_size,
                    self.square_size,
                    self.square_size
            ))

        self.rendered = True

    def draw():
        pygame.display.flip()
    
    def finished(self):
        return self.game_finished

    def screenshot(self):
        self.render()

        ret = []

        for y in range(self.field_size[1]):
            ret.append([self.border_line_color])

            for x in range(self.field_size[0]):
                if [x, y] in self.snake[:-1]:
                    ret[-1].append(self.snake_color)
                elif [x, y] == self.snake[-1]:
                    ret[-1].append(self.snake_head_color)
                elif [x, y] in self.food:
                    ret[-1].append(self.food_color)
                else:
                    ret[-1].append(self.background_color)

            ret[-1].append(self.border_line_color)

        ret = [[self.border_line_color for i in range(self.field_size[0] + 2)]] + ret + [[self.border_line_color for i in range(self.field_size[1] + 2)]]

        return np.array(ret, dtype=np.uint8)