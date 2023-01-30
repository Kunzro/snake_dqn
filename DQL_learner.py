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

"""This is an attempt to make a DQL that learns how to play Snake"""

import os
import time
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from collections import deque
import numpy as np

class DQL_learner:
    """This class creates and allows a NN to be trained based on Deep Q-Learning (DQL).
    Based on DQL means that I originally started with a generic DQL algorithm 
    and then started to add some modifications"""
    def __init__(self, game, input_shape, num_of_actions, batch_size, model_name, backup_timer = 50, target_update=30, gamma=0.95, memory_cap=100000, learning_rate=0.001,
                exploration_time=100, exploration_decay=0.99, exploration_rate_min=0.0001, save_directory="./saves", train_every_n_episodes = 1):
        self.model_name = model_name
        self.save_directory = save_directory
        self.backup_timer = backup_timer
        self.train_every_n_episodes = train_every_n_episodes
        self.episode_timer = 0
        self.gamma = gamma
        self.target_update = target_update
        self.target_update_time = 0
        self.batch_size = batch_size
        self.num_of_actions = num_of_actions
        self.game = game
        self.time = 0
        self.exploration_time = exploration_time
        self.exploration_decay = exploration_decay
        self.exploration_rate = 1
        self.exploration_rate_min = exploration_rate_min
        self.learning_rate = learning_rate
        self.memory_cap = memory_cap
        self.input_shape = input_shape
        self.memory = deque(maxlen=memory_cap)
        self.policy_NN = self._make_NN()
        self.target_NN = self._make_NN()
        self.target_NN.set_weights(self.policy_NN.get_weights())

    def _make_NN(self):
        """This function creates and returns a NN"""
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

        """
        game_board_input = keras.Input(shape=self.input_shape, name="board")
        x_conv = keras.layers.Conv2D(128, 3, activation="relu", kernel_regularizer=regularizers.l2(l=0.001), padding='same')(game_board_input)
        x_conv = keras.layers.Conv2D(64, 3, activation="relu", kernel_regularizer=regularizers.l2(l=0.001), padding='same')(x_conv)
        #two_dim = keras.layers.Matwo_dimPool2D(pool_size=(2, 2), strides=(1, 1))(two_dim)
        x_conv = keras.layers.Flatten()(x_conv)
        x_conv = keras.layers.Dense(16, activation="relu", kernel_regularizer=regularizers.l2(l=0.001))(x_conv)
        model_output = keras.layers.Dense(4, activation="linear", kernel_regularizer=regularizers.l2(l=0.001))(x_conv)
        model = keras.Model(inputs = game_board_input, outputs = model_output)
        model.summary()
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate), loss="mse")
        return model
        
        game_board_input = keras.Input(shape=self.input_shape[0], name="board")
        two_dim = keras.layers.Conv2D(256, 3, activation="relu", kernel_regularizer=regularizers.l2(l=0.0001), padding=)(game_board_input)
        two_dim = keras.layers.Conv2D(128, 3, activation="relu", kernel_regularizer=regularizers.l2(l=0.001))(two_dim)
        #two_dim = keras.layers.Matwo_dimPool2D(pool_size=(2, 2), strides=(1, 1))(two_dim)
        two_dim = keras.layers.Flatten()(two_dim)
        #two_dim = keras.layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(l=0.0001))(two_dim)

        one_dim_inp = keras.Input(shape=self.input_shape[1], name="head")
        one_dim = keras.layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(l=0.0001))(one_dim_inp)
        concatenated = keras.layers.Concatenate()([two_dim, one_dim])
        concatenated = keras.layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(l=0.001))(concatenated)
        model_output = keras.layers.Dense(4, activation="linear", kernel_regularizer=regularizers.l2(l=0.0001))(concatenated)
        model = keras.Model(inputs=[game_board_input, one_dim_inp], outputs=model_output)
        model.summary()
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate), loss="mse")
        return model
        """
 
        game_board_input = keras.Input(shape=self.input_shape, name="vision")
        dense = keras.layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(l=0.0001))(game_board_input)
        dense = keras.layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(l=0.0001))(dense)
        #dense = keras.layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(l=0.0001))(dense)
        model_output = keras.layers.Dense(4, activation="linear", kernel_regularizer=regularizers.l2(l=0.0001))(dense)
        model = keras.Model(inputs = [game_board_input], outputs = [model_output])
        model.summary()
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate), loss="mse")
        return model
        
        
    def _remember(self, *args):
        """This function is used to memorize an arbitraty ammount of arguments and unpacks tupels."""
        self.memory.append([arg for arg in args])

    def _replay(self):
        """This function loads training states from the momory."""
        if self.batch_size > len(self.memory):
            return
        else:
            samples = random.sample(self.memory, self.batch_size)
            return [[np.stack([row[i][j] for row in samples]) for j in range(len(samples[0][i]))] if isinstance(samples[0][i], list) else np.stack([row[i] for row in samples]) for i in range(len(samples[0]))]

    def _enought_memory(self):
        """This function checks wheter or not there are enough momorized training states for a batch."""
        return len(self.memory) >= self.batch_size


    def _select_action(self, state):
        """This function selects the action to be taken while simultaniously handling the exploration aspect."""
        self.time += 1
        if self.time > self.exploration_time and self.exploration_rate > self.exploration_rate_min:
            self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate*self.exploration_decay)

        if self.time < self.exploration_time or self.exploration_rate > random.random():
            print("random action chosen")
            return np.array(random.randint(0, self.num_of_actions-1))
        else:
            if isinstance(state, list):
                state = [np.stack([sub_state]) for sub_state in state]
            else:
                state = np.stack([state])
            predicted = self.policy_NN.predict(state)
            print("predicted move {}, selected {}".format(predicted, np.argmax(predicted)))
            return np.argmax(predicted)

    def load_save_state(self, name=None):
        """This function loads savepoints specified by its name and searches for the latest backup"""
        if name != None:
            savepoints = [self.save_directory + "/" + folder_name for folder_name in os.listdir(self.save_directory) if name in folder_name]
            if savepoints:
                directories = ".".join(savepoints)
                backup_num = max([int(s) for s in directories.split(".") if s.isdigit()])
                latest = [savepoint for savepoint in savepoints if str(backup_num) in savepoint]
                self.policy_NN = keras.models.load_model(latest[0])
                self.target_NN = keras.models.load_model(latest[0])
                self.episode_timer = backup_num * self.backup_timer
                self.exploration_rate = self.exploration_rate_min
                self.time = self.exploration_time + 1
                print("loaded save from {}".format(latest[0]))
            else:
                print("no name given or no backup found - training new model")
        else:
            print("no name given or no backup found - training new model")

    def episode(self):
        """This function trains for one episode (untill game_over = true)"""
        self.episode_timer += 1
        self.game.initialize()
        state = self.game.get_train_state()
        game_over = False
        not_trained = True
        # make backups of the NN every self.backup_timer episodes
        if self.episode_timer % self.backup_timer == 0:
            self.policy_NN.save(self.save_directory + "/" + self.model_name +  ".{}".format(int(self.episode_timer/self.backup_timer)) + ".h5")
        while not(game_over):
            # select actions to perform
            action = self._select_action(state)
            # emulate selected action
            reward, game_over, new_state = self.game.move(action)
            # store experience
            self._remember(state, action, reward, new_state, game_over)
            state = new_state

            if self._enought_memory() and self.exploration_time < self.time:
                #print(len(self.memory))
                if self.target_update_time % self.target_update == 0:
                    print("target NN updated")
                    self.target_NN.set_weights(self.policy_NN.get_weights())
                self.target_update_time += 1

                if not_trained and self.episode_timer % self.train_every_n_episodes == 0:
                    #not_trained = False
                    for i in range(1):
                        # load batch for training
                        states, actions, rewards, new_states, game_overs = self._replay()

                        y = self.policy_NN.predict(states)
                        a_star = np.argmax(self.policy_NN.predict(new_states), axis=1)
                        targets = np.where(game_overs, rewards, rewards + self.gamma * self.target_NN.predict(new_states)[range(new_states.shape[0]) ,a_star])
                        # limit the max predicted value to the max reward? (not clear if it really makes sense yet)
                        #targets = np.where(targets > 5, 5, targets)
                        y[range(y.shape[0]), actions] = targets
                    
                        # train the NN with the batch
                        self.policy_NN.fit(states, y, epochs=1, verbose=0, batch_size=self.batch_size)

    def play(self, model_name, delay_between_turns=0.1):
        """This function loads a model and plays the game without training it"""
        demonstration_NN = keras.models.load_model(self.save_directory + "/{}".format(model_name))
        self.game.initialize()
        game_over = False
        state = self.game.get_train_state()
        while not(game_over):
            if isinstance(state, list):
                state = [np.expand_dims(sub_state, axis=0) for sub_state in state]
            else:
                state = np.expand_dims(state, axis=0)
            action = np.argmax(demonstration_NN.predict(state))
            _, game_over, state = self.game.move(action)
            time.sleep(delay_between_turns)
