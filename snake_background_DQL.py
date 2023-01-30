    # Snake DQL
    # Copyright (C) 2020  Roman Kunz

    # This program is free software: you can redistribute it and/or modify
    # it under the terms of the GNU Affero General Public License as published
    # by the Free Software Foundation, either version 3 of the License, or
    # (at your option) any later version.

    # This program is distributed in the hope that it will be useful,
    # but WITHOUT ANY WARRANTY; without even the implied warranty of
    # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    # GNU Affero General Public License for more details.

    # You should have received a copy of the GNU Affero General Public License
    # along with this program.  If not, see <http://www.gnu.org/licenses/>.

    # contact: Roman Kunz: kunzro@student.ethz.ch

"""This Class impelenets the game logic using numpy and contains the required functions for the DQL_learner"""

import numpy as np
import pygame

REWARD_LOSE = -1
REWARD_APPLE = 1
REWARD_OKAY = 0

class Snake:
    """This class implements the game logic"""

    def __init__(self, width=20, height=20, window_width=400, window_height=400):
        self.death_timer = width*height*2
        self.width = width
        self.height = height
        self.game_board = np.zeros((height, width, 3), dtype=np.bool)
        head = (int(height/2), int(width/2))
        tail = (int(height/2), int(width/2)-1)
        self.game_board[head][0] = 1
        self.game_board[tail][1] = 1
        self.body_parts = [head, tail]
        self.direction = 0
        self.__spawn_apple()
        self.window_width = window_width
        self.window_height = window_height
        self.display = False
        self._update_display()

    def initialize(self):
        self.death_timer = self.width*self.height*2
        self.game_board = np.zeros((self.height, self.width, 3), dtype=np.int8)
        head = (int(self.height/2), int(self.width/2))
        tail = (int(self.height/2), int(self.width/2)-1)
        self.game_board[head][0] = 1
        self.game_board[tail][1] = 1
        self.body_parts = [head, tail]
        self.direction = 0
        self.__spawn_apple()

    def __spawn_apple(self):
        inds = np.where((self.game_board[:, :, 0] == 0) & (self.game_board[:, :, 1] == 0))
        #print("Number of possible places for apple: {}".format(inds[0].size))
        rand_int = np.random.randint(0, inds[0].size)
        position = (inds[0][rand_int], inds[1][rand_int], 2)
        self.game_board[position] = 1
        self.apple_pos = position[0:2]

    def __check_move(self, target_pos):
        """Checks what the result to the given move is."""
        if target_pos[0] < self.height and target_pos[1] < self.width and target_pos[0] >= 0 and target_pos[1] >= 0:
            if self.game_board[(target_pos[0], target_pos[1], 1)] == 0:
                if self.game_board[(target_pos[0], target_pos[1], 2)] == 1:
                    #print("apple eaten")
                    return "apple"
                #print("okay")
                return "okay"
        #print("GAME OVER")
        return "game_over"

    def _check_options(self):
        """This function checks what the results would be for all possible moves."""
        head_y, head_x = self.body_parts[0]
        all_move_pos = [(head_y, head_x+1), (head_y, head_x-1), (head_y-1, head_x), (head_y+1, head_x)]
        rewards = np.zeros((4))
        for counter, target_pos in enumerate(all_move_pos):
            rewards[counter] = REWARD_LOSE
            if target_pos[0] < self.height and target_pos[1] < self.width and target_pos[0] >= 0 and target_pos[1] >= 0:
                if self.game_board[(target_pos[0], target_pos[1], 1)] == 0:
                    if self.game_board[(target_pos[0], target_pos[1], 2)] == 1:
                        rewards[counter] = REWARD_APPLE
                    else:
                        rewards[counter] = REWARD_OKAY
        return rewards

    def _update_display(self):
        """This function draws the graphical representation of the game using pygame."""
        if not(self.display):
            pygame.init()
            WIN = pygame.display.set_mode((self.window_width, self.window_height))
            pygame.display.set_caption("Snake")

        WIN.fill(((0, 0, 0)))
        for column in range(self.width+1):
            x = column*self.window_width/self.width
            y = 0
            pygame.draw.rect(WIN, (128, 128, 128), (x-2, y, 4, self.window_height))

        for row in range(self.window_height+1):
            x = 0
            y = row*self.window_height/self.height
            pygame.draw.rect(WIN, (128, 128, 128), (x, y-2, self.window_width, 4))

        square_x = self.window_width/self.width-4
        square_y = self.window_height/self.height-4

        offset_x = self.window_width/self.width
        offset_y = self.window_height/self.height

        tail = np.where(self.game_board[:, :, 1] == 1)
        #apple = np.where(self.game_board[:, :, 2] == 1)
        #head = np.where(self.game_board[:, :, 0] == 1)

        pygame.draw.rect(WIN, (255, 0, 0), pygame.Rect(self.apple_pos[1]*offset_x+2, self.apple_pos[0]*offset_y+2, square_x, square_y))
        pygame.draw.rect(WIN, (0, 255, 0), pygame.Rect(self.body_parts[0][1]*offset_x+2, self.body_parts[0][0]*offset_y+2, square_x, square_y))

        for box in np.transpose(np.array(tail)):
            pygame.draw.rect(WIN, (0, 128, 0), (box[1]*offset_x+2, box[0]*offset_y+2, square_x, square_y))

        pygame.display.update()

    def vision(self):
        """this function produces training states that are based on what the snake sees when looking in different directions as well as other features"""
        y_head, x_head = self.body_parts[0]
        body_board = self.game_board[:, :, 0] + self.game_board[:, :, 1]
        apple_board = self.game_board[:, :, 2]

        # up
        up = np.zeros((3), np.int8)
        distance = 1
        x, y = x_head, y_head
        while y > 0:
            y -= 1
            if up[0] == 0 and body_board[y, x] == 1:
                up[0] = distance
            if up[1] == 0 and apple_board[y, x] == 1:
                up[1] = distance
            up[2] = distance+1
            distance += 1

        # down
        down = np.zeros((3), np.int8)
        distance = 1
        x, y = x_head, y_head
        while y < self.height-1:
            y += 1
            if down[0] == 0 and body_board[y, x] == 1:
                down[0] = distance
            if down[1] == 0 and apple_board[y, x] == 1:
                down[1] = distance
            down[2] = distance+1
            distance += 1

        # left
        left = np.zeros((3), np.int8)
        distance = 1
        x, y = x_head, y_head
        while x > 0:
            x -= 1
            if left[0] == 0 and body_board[y, x] == 1:
                left[0] = distance
            if left[1] == 0 and apple_board[y, x] == 1:
                left[1] = distance
            left[2] = distance+1
            distance += 1

        # right
        right = np.zeros((3), np.int8)
        distance = 1
        x, y = x_head, y_head
        while x < self.width-1:
            x += 1
            if right[0] == 0 and body_board[y, x] == 1:
                right[0] = distance
            if right[1] == 0 and apple_board[y, x] == 1:
                right[1] = distance
            right[2] = distance+1
            distance += 1

        # up-right
        up_right = np.zeros((3), np.int8)
        distance = 1
        x, y = x_head, y_head
        while x < self.width-1 and y > 0:
            x += 1
            y -= 1
            if up_right[0] == 0 and body_board[y, x] == 1:
                up_right[0] = distance
            if up_right[1] == 0 and apple_board[y, x] == 1:
                up_right[1] = distance
            up_right[2] = distance+1
            distance += 1

        # up-left
        up_left = np.zeros((3), np.int8)
        distance = 1
        x, y = x_head, y_head
        while x > 0 and y > 0:
            x -= 1
            y -= 1
            if up_left[0] == 0 and body_board[y, x] == 1:
                up_left[0] = distance
            if up_left[1] == 0 and apple_board[y, x] == 1:
                up_left[1] = distance
            up_left[2] = distance+1
            distance += 1

        # down-right
        down_right = np.zeros((3), np.int8)
        distance = 1
        x, y = x_head, y_head
        while x < self.width-1 and y < self.height-1:
            x += 1
            y += 1
            if down_right[0] == 0 and body_board[y, x] == 1:
                down_right[0] = distance
            if down_right[1] == 0 and apple_board[y, x] == 1:
                down_right[1] = distance
            down_right[2] = distance+1
            distance += 1

        # down-left
        down_left = np.zeros((3), np.int8)
        distance = 1
        x, y = x_head, y_head
        while x > 0 and y < self.height-1:
            x -= 1
            y += 1
            if down_left[0] == 0 and body_board[y, x] == 1:
                down_left[0] = distance
            if down_left[1] == 0 and apple_board[y, x] == 1:
                down_left[1] = distance
            down_left[2] = distance+1
            distance += 1

        # length
        length = np.array([len(self.body_parts)])

        # distnance to apple (one-norm)
        y_apple, x_apple = self.apple_pos
        distance_apple = abs((y_apple - y_head)) + abs((x_apple - x_head))

        # direction of apple
        direction_apple = np.zeros((4), np.int8)
        y_apple, x_apple = self.apple_pos
        y_diff, x_diff = y_apple - y_head, x_apple - x_head
        if abs(y_diff) > abs(x_diff):
            if y > 0:
                direction_apple[0] = 1
            else:
                direction_apple[1] = 1
        else:
            if x > 0:
                direction_apple[2] = 1
            else:
                direction_apple[3] = 1

        return np.concatenate((direction_apple, up, down, left, right))


    def board_head_apple_direction(self):
        """This is the function that is called when the state for training is requested."""
        game_board = self.game_board[:, :, 0] + self.game_board[:, :, 1]
        head_pos = np.array(self.body_parts[0])
        apple_pos = self.apple_pos[0:2]

        # direction of apple
        y_head, x_head = self.body_parts[0]
        direction_apple = np.zeros((4), np.int8)
        y_apple, x_apple = self.apple_pos[0:2]
        y_diff, x_diff = y_apple - y_head, x_apple - x_head
        if abs(y_diff) > abs(x_diff):
            if y_diff > 0:
                direction_apple[0] = 1
            else:
                direction_apple[1] = 1
        else:
            if x_diff > 0:
                direction_apple[2] = 1
            else:
                direction_apple[3] = 1

        one_dim = np.concatenate((direction_apple, head_pos, apple_pos))

        return [game_board, one_dim]
    
    def get_train_state(self):
        #return self.board_head_apple_direction()
        return self.vision()
        #return self.game_board

    def get_state(self):
        """This function returns the entire state of the game, such that it can be saved."""
        return (self.game_board, self.body_parts, self.direction)

    def set_state(self, state):
        """This function resets the state of the game to the state given."""
        self.game_board, self.body_parts, self.direction = state

    def reward(self):
        """This is an attempt to make a better reward function other than pure constants."""
        y_head, x_head = self.body_parts[0]
        y_apple, x_apple = np.where(self.game_board[:, :, 2] == 1)

        if self.direction == 0:
            if x_apple < x_head:
                return 0.2

        if self.direction == 1:
            if x_apple > x_head:
                return 0.2

        if self.direction == 2:
            if y_apple < y_head:
                return 0.2

        if self.direction == 3:
            if y_apple > y_head:
                return 0.2

        return -0.2

    def move(self, direction=None):
        """ This function makes a given move and returns the reward, game_over and the training_state.
            The reward represents wheter the action was good bad or indifferent.
            Game_over is a bool variable and signals wheter the game is over or not.
            The training_state is what the controller can use to make further game decisions.
            """
        #all_options_results = self._check_options()

        if direction is None:
            direction = self.direction
        self.direction = direction

        # if death_timer is 0 start making random moves
        if self.death_timer <= 0:
            direction = np.random.randint(0, 4)
            print("starting to make random moves now")

        # right
        if direction == 0:
            target_pos = (self.body_parts[0][0], self.body_parts[0][1]+1)
        # left
        if direction == 1:
            target_pos = (self.body_parts[0][0], self.body_parts[0][1]-1)
        # up
        if direction == 2:
            target_pos = (self.body_parts[0][0]-1, self.body_parts[0][1])
        # down
        if direction == 3:
            target_pos = (self.body_parts[0][0]+1, self.body_parts[0][1])

        if len(self.body_parts) > 2:
            self.game_board[:, :, 1][self.body_parts[-1]] = 0
        event = self.__check_move(target_pos=target_pos)
        if len(self.body_parts) == 2:
            self.game_board[:, :, 1][self.body_parts[-1]] = 0

        if event == "game_over":
            self._update_display()
            return REWARD_LOSE, True, self.get_train_state()

        if event == "okay":
            self.death_timer -= 1
            self.game_board[:, :, 0][target_pos] = 1
            self.game_board[:, :, 0][self.body_parts[0]] = 0
            self.game_board[:, :, 1][self.body_parts[0]] = 1
            self.body_parts.insert(0, target_pos)
            self.body_parts.pop(-1)
            self._update_display()
            return self.reward(), False, self.get_train_state()

        if event == "apple":
            self.death_timer = self.width*self.height*2
            self.game_board[:, :, 1][self.body_parts[-1]] = 1
            self.game_board[:, :, 0][target_pos] = 1
            self.game_board[:, :, 0][self.body_parts[0]] = 0
            self.game_board[:, :, 1][self.body_parts[0]] = 1
            self.game_board[:, :, 2][target_pos] = 0
            self.body_parts.insert(0, target_pos)
            self.__spawn_apple()
            self._update_display()
            return REWARD_APPLE, False, self.get_train_state()
