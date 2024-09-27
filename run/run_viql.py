import numpy as np
import logging
from bidding_train_env.common.utils import normalize_state, normalize_reward, save_normalize_dict
from bidding_train_env.baseline.viql.buffer import ReplayBuffer
from bidding_train_env.baseline.viql.viql import IQL
import sys
import pandas as pd
import ast
import random
import torch

np.set_printoptions(suppress=True, precision=4)
logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

STATE_DIM = 16
ACTION_DIM = 1
learning_rate = 3e-4
hidden_size = 256
tau = 5e-1
temperature = 3
expectile = 0.6

def reward_consider_exceed(training_data):
    beta = 2
    coef = (training_data['CPAConstraint'] * training_data['reward_continuous']) / (training_data['realTimeCost'] + 1e-10)
    coef = coef * training_data['CPAConstraint'] / 100
    penalty = np.minimum(1,pow(coef, beta))
    penalty2 = np.maximum(0.8 - coef, 0) * 100 + np.maximum(0.93 - coef, 0) * 30 # 超成本：coef小于1

    training_data['reward_exceed'] = penalty * training_data['reward_continuous'] - penalty2 # * training_data['CPAConstraint']

def train_viql_model():
    """
    Train the IQL model.
    """
    config_seed = 2024
    np.random.seed(config_seed)
    random.seed(config_seed)
    torch.manual_seed(config_seed)

    train_data_path = "./data/traffic/training_data_rlData_folder/training_data_all-rlData.csv"
    training_data = pd.read_csv(train_data_path)

    def safe_literal_eval(val):
        if pd.isna(val):
            return val
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            print(ValueError)
            return val

    training_data["state"] = training_data["state"].apply(safe_literal_eval)
    training_data["next_state"] = training_data["next_state"].apply(safe_literal_eval)
    reward_consider_exceed(training_data)
    is_normalize = True

    if is_normalize:
        normalize_dic = normalize_state(training_data, STATE_DIM, normalize_indices=[13, 14, 15])
        # select use continuous reward
        #training_data['reward'] = normalize_reward(training_data, "reward_continuous")
        training_data['reward'] = normalize_reward(training_data, "reward_exceed")
        # select use sparse reward
        # training_data['reward'] = normalize_reward(training_data, "reward")
        save_normalize_dict(normalize_dic, "saved_model/VIQLtest")

    # Build replay buffer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    replay_buffer = ReplayBuffer(2000000, 100, device)
    add_to_replay_buffer(replay_buffer, training_data, is_normalize)
    print(len(replay_buffer.memory))

    # Train model
    model = IQL(STATE_DIM, ACTION_DIM, learning_rate, hidden_size, tau, temperature, expectile, device)
    train_model_steps(model, replay_buffer)

    # Save model
    model.save_jit("saved_model/VIQLtest")

    # Test trained model
    test_trained_model(model, replay_buffer)


def add_to_replay_buffer(replay_buffer, training_data, is_normalize):
    for row in training_data.itertuples():
        state, action, reward, next_state, done = row.state if not is_normalize else row.normalize_state, row.action, row.reward if not is_normalize else row.normalize_reward, row.next_state if not is_normalize else row.normalize_nextstate, row.done
        # ! 去掉了所有的done==1的数据
        if done != 1:
            replay_buffer.add(np.array(state), np.array([action]), np.array([reward]), np.array(next_state),
                               np.array([done]))
        else:
            replay_buffer.add(np.array(state), np.array([action]), np.array([reward]), np.zeros_like(state),
                               np.array([done]))


def train_model_steps(model, replay_buffer, step_num=20000, batch_size=100):
    for i in range(step_num):
        states, actions, rewards, next_states, terminals = replay_buffer.sample()
        experi = [ states, actions, rewards, next_states, terminals ]
        a_loss, c1_loss, c2_loss, v_loss = model.learn(experi)
        if i % 500 == 0:
            logger.info(f'Step: {i} c1_loss: {c1_loss} c2_loss: {c2_loss} V_loss: {v_loss} A_loss: {a_loss}')


def test_trained_model(model, replay_buffer):
    states, actions, rewards, next_states, terminals = replay_buffer.sample()
    pred_actions = model.get_action(states, True)
    actions = actions.cpu().detach().numpy()
    tem = np.concatenate((actions, pred_actions), axis=1)
    print("action VS pred action:", tem)


def run_viql():
    print(sys.path)
    """
    Run IQL model training and evaluation.
    """
    train_viql_model()


if __name__ == '__main__':
    run_viql()
