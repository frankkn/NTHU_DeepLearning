import copy
import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf

_EVAL_DIR = 'evaluation'
_USER_FEATURES = os.path.join(_EVAL_DIR, 'user_features.pkl')
_ITEM_FEATURES = os.path.join(_EVAL_DIR, 'item_features.pkl')

_N_TRAINING_USERS = 1000
_N_USERS = 2000
_N_ITEMS = 209527
_EMBEDDING_DIM = 42

_SLATE_SIZE = 5
_HORIZON = 2000
_NO_CLICK_THRESHOLD = 1.7
_SOFTMAX_TEMPERATURE = 0.1
_INIT_PATIENCE = 5.0
_PATIENCE_INCREMENT_WEIGHT = 0.5

class SimulationEnvironment:
    '''Base class for training and testing environments.'''

    def __init__(self, n_users: int) -> None:
        '''
        Initialize a simulation environment with `n_users` users.

        Args:
            n_users: `int`
                The total number of users in the simulation environment.
        '''
        self._n_users = n_users
        self._active_users = [i for i in range(self._n_users)]
        self._user_patience = [_INIT_PATIENCE for _ in range(self._n_users)]
        self._user_session_length = [0 for _ in range(self._n_users)]

        self._df_user = pd.read_pickle(_USER_FEATURES)
        self._df_user = self._df_user[:self._n_users]
        self._user_history = [copy.deepcopy(h) for h in self._df_user['history']]
        self._user_embeddings = tf.convert_to_tensor([x for x in self._df_user['embedding']], dtype=tf.float32)

        df_item = pd.read_pickle(_ITEM_FEATURES)
        self._item_embeddings = tf.convert_to_tensor([x for x in df_item['embedding']], dtype=tf.float32)

        self._cur_user = random.choice(self._active_users)


    def reset(self) -> None:
        '''Reset the environment to its initial parameters and states.'''
        self._active_users = [i for i in range(self._n_users)]
        self._user_patience = [_INIT_PATIENCE for _ in range(self._n_users)]
        self._user_session_length = [0 for _ in range(self._n_users)]
        self._user_history = [copy.deepcopy(h) for h in self._df_user['history']]
        self._cur_user = random.choice(self._active_users)


    def has_next_state(self) -> bool:
        '''
        Verify whether the next state exists.
        The next state is considered to exist if there is at least one user still present in the environment.

        Returns:
            `True` if the next state exists, `False` otherwise.
        '''
        return len(self._active_users) != 0


    def get_state(self) -> int:
        '''
        Get the current state (the user ID of the current user).

        Returns:
            `int`: The user ID of the current user, or `-1` if there are no active users in the environment.
        '''
        return self._cur_user


    def get_response(self, slate: list[int]) -> tuple[int, bool]:
        '''
        Send the recommended slate (list of 5 distinct item IDs) and get the response from the current user.
        The internal user state will be updated according to the response, and a random user will be selected to be the next user (next state).

        Args:
            slate: `list[int]`
                A list of 5 distinct item IDs to be recommended.

        Returns:
            `tuple[int, bool]`:
                The first entry indicates the `item ID` chosen by the user, or `-1` if the user decides not to choose any item.
                The second entry represents whether the user is still in the environment after this interaction round. `True` if the user stays, `False` if the user leaves.

        Raises:
            `AssertionError`: If the slate length is not 5, contains duplicates or out-of-range item IDs, or if there are no active users in the environment.
        '''
        slate = [int(i) for i in slate]
        assert len(np.unique(slate)) == _SLATE_SIZE
        assert min(slate) >= 0 and max(slate) < _N_ITEMS
        assert self._cur_user != -1

        user_emb = self._user_embeddings[self._cur_user]
        item_embs = tf.gather(self._item_embeddings, indices=slate)
        sim_score = tf.squeeze(tf.matmul(item_embs, tf.reshape(user_emb, shape=(-1, 1)))).numpy().tolist()

        for i, id in enumerate(slate):
            if id in self._user_history[self._cur_user]:
                sim_score[i] = tf.float32.min

        sim_score.append(_NO_CLICK_THRESHOLD)
        probs = tf.nn.softmax(tf.convert_to_tensor(sim_score) / _SOFTMAX_TEMPERATURE)
        log_probs = tf.math.log(probs)
        choice = tf.squeeze(tf.random.categorical([log_probs], num_samples=1)).numpy()

        chosen_item = -1
        user_stays = True
        self._user_session_length[self._cur_user] += 1

        if choice == _SLATE_SIZE:
            chosen_item = -1
            self._user_patience[self._cur_user] -= 1.
        else:
            chosen_item = slate[choice]
            self._user_history[self._cur_user].append(chosen_item)
            self._user_patience[self._cur_user] += min(max(probs[choice].numpy() * _PATIENCE_INCREMENT_WEIGHT, 0.0), 1.0)

        if self._user_patience[self._cur_user] <= 0. or self._user_session_length[self._cur_user] >= _HORIZON:
            user_stays = False
            self._active_users.remove(self._cur_user)

        if self.has_next_state() and len(self._active_users) > 0:
            self._cur_user = random.choice(self._active_users)
        else:
            self._cur_user = -1

        return chosen_item, user_stays


    def get_score(self) -> list[float]:
        '''
        Get the normalized session length score (0 ~ 1) for each user.

        Returns:
            `list[float]`: A list containing the normalized session length score for each user.
        '''
        return [s / _HORIZON for s in self._user_session_length]


class TrainingEnvironment(SimulationEnvironment):
    '''
    Class for the training environment. Contains first 1000 users with user ID ranging from 0 to 999.
    '''

    def __init__(self) -> None:
        super().__init__(n_users=_N_TRAINING_USERS)


class TestingEnvironment(SimulationEnvironment):
    '''
    Class for the testing environment. Contains all 2000 users with user ID ranging from 0 to 1999.
    '''

    def __init__(self) -> None:
        super().__init__(n_users=_N_USERS)
