"""
    Author: Zhanluo Zhang
    Author E-mail: zhangzhanluo@outlook.com
    Version: v1.0
    Created Date: 20220627
    Description: experiments
"""
import random
import time
import os
import numpy as np
import torch
from code_020_network import ACTSchedulingNetworkSolver
from code_030_rl import DQN, ReplayBuffer, add_data, add_log
from code_011_instances import training_instance_settings

lr = 1e-4
num_episodes = 50000
hidden_dim = 128
gamma = 0.98
epsilon = 0.05
target_update = 100
buffer_size = 10000
minimal_size = 1000
batch_size = 100
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

# 设定存储路径
data_save_path = '01_Data/{}/'.format(time.strftime('%m-%d %H-%M'))
if not os.path.exists(data_save_path):
    os.mkdir(data_save_path)

# 设定存储文件
log_path = data_save_path + 'log.log'
data_path = data_save_path + 'data.csv'

best_scores = [1e10 for _ in range(len(training_instance_settings))]
actions = ['min-degree', 'max-degree', 'min-processing_time', 'max-processing_time',
           'min-collective_influence', 'max-mapping_entropy']
with open(log_path, 'w', encoding='utf-8') as f:
    f.write('For IEEM\n\n')  # notes in log file
    f.write('Instance Setting: ' + str(training_instance_settings) + '\n')
    f.write('\nActions: ' + str(actions) + '\n')
    f.write('\nEpisodes: ' + str(num_episodes) + '\n')
with open(data_path, 'w', encoding='utf-8') as f:
    f.write('env_id,makespan,return\n')
attrs = list(set([action.split('-')[1] for action in actions]))
attrs.sort()
envs = [ACTSchedulingNetworkSolver(**instance_setting) for instance_setting in training_instance_settings]
for i, env in enumerate(envs):
    add_log(log_path, '\n\nInstance {}\n'.format(i))
    makespans = 0
    random_k = 10
    for _ in range(random_k):
        scheduling_result = env.scheduling()
        makespans += scheduling_result['makespan']
    add_log(log_path, '\nRandom方法调度makespan平均为{:.2f}s\n'.format(makespans / random_k))
    for action in actions:
        scheduling_strategy, scheduling_property = action.split('-')
        scheduling_result = env.scheduling(scheduling_property, scheduling_strategy)
        add_log(log_path, '{}方法调度makespan为{:.2f}s\n'.format(action, scheduling_result['makespan']))
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
    env_id = random.randrange(0, len(envs))
    env = envs[env_id]
    episode_return = 0
    available_nodes = env.reset()
    state = env.cal_state(available_nodes, attrs)
    done = False
    while not done:
        action_idx = agent.take_action(state)
        available_nodes, reward, done = env.step(actions[action_idx])
        if done:
            makespan_list.append(env.clock)
            if env.clock < best_scores[env_id]:
                best_scores[env_id] = env.clock
                torch.save(agent.q_net.state_dict(),
                           data_save_path + 'best_model_weight_{}_{}_{:.1f}.pth'.format(env_id, i_episode, env.clock))
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
    add_data(data_path, [env_id, makespan_list[-1], return_list[-1]])
    if i_episode % 500 == 0:
        torch.save(agent.q_net.state_dict(), data_save_path + 'model_weight_{}.pth'.format(i_episode))
        add_log(log_path, '\n{}回合，算例{}，任务调度时间为{:.2f}秒\n'.format(i_episode, env_id, makespan_list[-1]))
