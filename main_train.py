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

import os
from DQL_learner import DQL_learner
from snake_background_DQL import Snake

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

x_size = 8
y_size = 8

model_name = "windows_vision2"

game = Snake(x_size, y_size)
DQL = DQL_learner(game, 16, 4, 64, model_name=model_name)

DQL.load_save_state(model_name)

while True:
    DQL.episode()
