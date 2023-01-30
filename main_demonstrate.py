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

"""Run this file to watch the specified model located in saves play a game of Snake"""

import time
from tensorflow import keras
import numpy as np
from snake_background_DQL import Snake

NETWORK_NAME = "vision_only-4.h5"
SAVE_DIR = "./saves/"
DELAY_BETWEEN_TURNS = 0.05

X_SIZE = 8
Y_SIZE = 8

MODEL = keras.models.load_model(SAVE_DIR+NETWORK_NAME)

GAME = Snake(X_SIZE, Y_SIZE)

GAME_OVER = False

while not GAME_OVER:
    state = GAME.get_train_state()
    if isinstance(state, list):
        state = [np.expand_dims(sub_state, axis=0) for sub_state in state]
    else:
        state = np.expand_dims(state, axis=0)
    action = np.argmax(MODEL.predict(state))
    _, GAME_OVER, state = GAME.move(action)
    time.sleep(DELAY_BETWEEN_TURNS)