"""
    Author: Zhanluo Zhang
    Author E-mail: zhangzhanluo@outlook.com
    Version: v1.0
    Created Date: 20220528
    Description: reinforcement learning
"""

import random
import time
import os
import numpy as np
import collections
import torch
import torch.nn.functional as F
from code_020_network import ACTSchedulingNetworkSolver


class ReplayBuffer:
    """ 经验回放池 """

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, _state, _action, _reward, _next_state, _done):  # 将数据加入buffer
        self.buffer.append((_state, _action, _reward, _next_state, _done))

    def sample(self, _batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, _batch_size)
        _state, _action, _reward, _next_state, _done = zip(*transitions)
        return np.array(_state), _action, _reward, np.array(_next_state), _done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)


class Qnet(torch.nn.Module):
    """ 只有一层隐藏层的Q网络 """

    def __init__(self, _state_dim, _hidden_dim, _action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(_state_dim, _hidden_dim)
        self.fc2 = torch.nn.Linear(_hidden_dim, _action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        return self.fc2(x)


class DQN:
    """ DQN算法 """

    def __init__(self, _state_dim, _hidden_dim, _action_dim, _learning_rate, _gamma,
                 _epsilon, _target_update, _device):
        self.action_dim = _action_dim
        self.q_net = Qnet(_state_dim, _hidden_dim,
                          self.action_dim).to(_device)  # Q网络
        # 目标网络
        self.target_q_net = Qnet(_state_dim, _hidden_dim,
                                 self.action_dim).to(_device)
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=_learning_rate)
        self.gamma = _gamma  # 折扣因子
        self.epsilon = _epsilon  # epsilon-贪婪策略
        self.target_update = _target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = _device

    def take_action(self, _state, episode=None):  # epsilon-贪婪策略采取动作
        if episode is not None:
            _epsilon = 100 / (episode + 1)
        else:
            _epsilon = self.epsilon
        if np.random.random() < _epsilon:
            _action = np.random.randint(self.action_dim)
        else:
            _state = torch.tensor([_state], dtype=torch.float).to(self.device)
            _action = self.q_net(_state).argmax().item()
        return _action

    def update(self, _transition_dict):
        states = torch.tensor(_transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(_transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(_transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(_transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(_transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)  # Q值
        # 下个状态的最大Q值
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(
            -1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 更新目标网络
        self.count += 1


def add_log(_log_path, _text):
    with open(_log_path, 'a', encoding='utf-8') as _f:
        _f.write(_text)


def add_data(_log_path, _data):
    _data = [str(round(x, 2)) for x in _data]
    with open(_log_path, 'a', encoding='utf-8') as _f:
        _f.write(','.join(_data))
        _f.write(','.join('\n'))


if __name__ == '__main__':
    lr = 2e-3
    num_episodes = 2000000
    hidden_dim = 128
    gamma = 0.98
    epsilon = 0.1
    target_update = 10
    buffer_size = 5000
    minimal_size = 500
    batch_size = 64
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")
    data_save_path = '01_Data/{}/'.format(time.strftime('%m-%d %H-%M'))
    log_path = data_save_path + 'log.log'
    data_path = data_save_path + 'data.csv'
    if not os.path.exists(data_save_path):
        os.mkdir(data_save_path)
    instance_setting = {'n_mts': 2, 'n_tps_per_mt': 2, 'n_agvs': 10, 'n_mates_per_asc': 4, 'n_ascs': 3,
                        'n_jobs_disc': 50, 'n_jobs_load': 50, 'job_mt': None, 'job_asc': None, 'random_seed': 2}
    actions = ['min-degree', 'max-degree', 'min-processing_time', 'max-processing_time',
               'min-collective_influence', 'max-mapping_entropy']
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('每次奖励0,回合奖励为-最大完工时间，epsilon采用100/(t+1)的策略，状态增加阶段比例\n\n')  # 日志里的备注信息
        f.write('Instance Setting: ' + str(instance_setting) + '\n')
        f.write('\nActions: ' + str(actions) + '\n')
        f.write('\nEpisodes: ' + str(num_episodes) + '\n')
    with open(data_path, 'w', encoding='utf-8') as f:
        f.write('makespan,return\n')
    attrs = list(set([action.split('-')[1] for action in actions]))
    attrs.sort()
    env = ACTSchedulingNetworkSolver(**instance_setting)
    random_k = 10
    makespans = 0
    for i in range(random_k):
        scheduling_result = env.scheduling()
        makespans += scheduling_result['makespan']
    add_log(log_path, '\nRandom方法调度makespan平均为{:.2f}s\n'.format(makespans / random_k))
    for action in actions:
        scheduling_strategy, scheduling_property = action.split('-')
        scheduling_result = env.scheduling(scheduling_property, scheduling_strategy)
        add_log(log_path, '{}方法调度makespan平均为{:.2f}s\n'.format(action, scheduling_result['makespan']))
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    replay_buffer = ReplayBuffer(buffer_size)
    state_dim = len(attrs) * len(attrs)
    action_dim = len(actions)
    agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
                target_update, device)

    return_list = []
    makespan_list = []

    for i_episode in range(num_episodes):
        episode_return = 0
        available_nodes = env.reset()
        state = env.cal_state(available_nodes, attrs)
        done = False
        while not done:
            action_idx = agent.take_action(state)
            available_nodes, reward, done = env.step(actions[action_idx])
            if done:
                makespan_list.append(env.clock)
            next_state = env.cal_state(available_nodes, attrs)
            replay_buffer.add(state, action_idx, reward, next_state, done)
            state = next_state
            episode_return += reward
            # 当buffer数据的数量超过一定值后,才进行Q网络训练
            if replay_buffer.size() > minimal_size:
                b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                transition_dict = {
                    'states': b_s,
                    'actions': b_a,
                    'next_states': b_ns,
                    'rewards': b_r,
                    'dones': b_d
                }
                agent.update(transition_dict)
        return_list.append(episode_return)
        add_data(data_path, [makespan_list[-1], return_list[-1]])
        if i_episode % 500 == 0:
            torch.save(agent.q_net.state_dict(), data_save_path + 'model_weight_{}.pth'.format(i_episode))
            add_log(log_path, '\n{}回合，任务调度时间为{:.2f}秒\n'.format(i_episode, makespan_list[-1]))
