"""
    Author: Zhanluo Zhang
    Author E-mail: zhangzhanluo@outlook.com
    Version: v1.0
    Created Date: 20210829
    Description: scheduling network and network scheduling
"""
import copy
import random
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

import code_010_preparation


class ACTSchedulingNetwork(code_010_preparation.Instance):
    def __init__(self, **_instance_setting):
        super(ACTSchedulingNetwork, self).__init__(**_instance_setting)
        self.stage_branches = [self.n_mts, self.n_mts, self.n_mts, 1, self.n_ascs, self.n_ascs]
        self.stage_branch_machines = [1, self.n_tps_per_mt, 1, self.n_agvs, self.n_mates_per_asc, 1]
        self.task_network = nx.DiGraph()
        self.task_nodes = []  # self.task_nodes[job_id][stage_id][branch_id][machine_id] -> node_id
        self.machine_network = nx.DiGraph()
        self.task_network_copy = None
        self.machine_nodes = []  # self.machine_nodes[stage_id][branch_id][machine_id] -> node_id
        self._build_task_network()
        self.task_network_copy = self.task_network.copy()
        self._build_machine_network()

    def _add_task_nodes(self):
        """
        添加任务节点。

        :return: 无
        """
        node_id = 0

        for i in range(self.n_jobs):
            stage_nodes = []
            for j in range(self.n_stages):
                branch_nodes = []
                for k in range(self.stage_branches[j]):
                    machine_nodes = []
                    for m in range(self.stage_branch_machines[j]):
                        if j in [0, 1, 2] and k != self.job_mt[i]:
                            continue  # 指定任务的主小车阶段编号
                        elif j in [4, 5] and k != self.job_asc[i]:
                            continue  # 指定任务的轨道吊阶段编号
                        elif j in [1, 4] and i < self.n_jobs_disc and m >= round(self.stage_branch_machines[j] / 2):
                            # 注意此处不能简单地跳过，分不清使用了哪一个机器
                            machine_nodes.append(-1)  # 卸船任务，跳过后半部分机器，并以-1计入到列表中（中转平台进行装卸专用指定）
                        elif j in [1, 4] and i >= self.n_jobs_disc and m < round(self.stage_branch_machines[j] / 2):
                            machine_nodes.append(-1)  # 装船任务，跳过前半部分机器，并以-1计入到列表中（意图同上）
                        else:
                            self.task_network.add_node(node_id, job_id=i, stage_id=j, branch_id=k, machine_id=m,
                                                       processing_time=self.processing_time[i, j])
                            machine_nodes.append(node_id)
                            node_id += 1
                    branch_nodes.append(machine_nodes)
                stage_nodes.append(branch_nodes)
            self.task_nodes.append(stage_nodes)

        # 遍历节点，添加方向信息
        for node in self.task_network.nodes:
            # 添加任务方向信息
            self.task_network.nodes[node]['direction'] = 1 if self.task_network.nodes[node][
                                                                  'job_id'] < self.n_jobs_disc else -1
        # 对所有节点添加排他节点信息(同阶段同分支其他机器的节点)
        for i in range(self.n_jobs):
            for j in range(self.n_stages):
                for k in range(self.stage_branches[j]):
                    for m in range(self.stage_branch_machines[j]):
                        # 同阶段同分支其他机器的节点
                        same_branch_different_machine_nodes = copy.deepcopy(self.task_nodes[i][j][k])
                        if len(same_branch_different_machine_nodes) > 0:  # 对于非指定的起止点分支，这个长空为空，没有有效的机器
                            if same_branch_different_machine_nodes[m] != -1:  # 如果机器节点取-1，这意味着该机器并不适用于该任务
                                same_branch_different_machine_nodes.pop(m)  # 去除自身
                                same_branch_different_machine_nodes = [x for x in same_branch_different_machine_nodes if
                                                                       x != -1]  # 无效的机器也不用加入
                                self.task_network.nodes[self.task_nodes[i][j][k][m]][
                                    'exclude_nodes'] = same_branch_different_machine_nodes

    def _add_task_edges(self):
        """
        添加任务网络连边。

        :return: 无。
        """

        for i in range(self.n_jobs):
            for j in range(self.n_stages):
                for k in range(self.stage_branches[j]):
                    if len(self.task_nodes[i][j][k]) > 0:
                        for m in range(self.stage_branch_machines[j]):
                            node_id = self.task_nodes[i][j][k][m]
                            if node_id == -1:
                                continue
                            # 阶段连边，同一任务在同一类型机器上仅加工一次，其他分支和其他机器不再工作
                            for h in range(self.stage_branch_machines[j]):
                                if h != m and self.task_nodes[i][j][k][m] != -1 and self.task_nodes[i][j][k][h] != -1:
                                    self.task_network.add_edge(node_id, self.task_nodes[i][j][k][h],
                                                               type='Same Job Same Phase')
                            # 资源连边，同一机器不会同时处理多个任务，双向，此处是i->n
                            for n in range(self.n_jobs):
                                if len(self.task_nodes[n][j][k]) > 0 and n != i and self.task_nodes[n][j][k][m] != -1:
                                    self.task_network.add_edge(node_id, self.task_nodes[n][j][k][m],
                                                               type='Same Machine',
                                                               weight=self.setup_time[i][n][j])
                            # 任务连边。任务具有先后顺序，单向
                            direction = self.task_network.nodes[self.task_nodes[i][j][k][m]]['direction']
                            if j < self.n_stages - 1:
                                if j in [0, 1, 4]:
                                    disc_downstream_nodes = self.task_nodes[i][j + 1][k]
                                else:
                                    disc_downstream_nodes = sum(self.task_nodes[i][j + 1], [])
                                for node in disc_downstream_nodes:
                                    if node == -1:
                                        continue
                                    if direction == 1:
                                        self.task_network.add_edge(node_id, node, type='Task Order')
                                    else:
                                        self.task_network.add_edge(node, node_id, type='Task Order')

    def _build_task_network(self):
        """
        建立任务网络。包括建立节点和连边。

        :return: 无
        """
        self._add_task_nodes()
        self._add_task_edges()

    def _get_all_branch_machines(self, stage_id):
        """
        获得同阶段所有分支的机器。

        :param stage_id: 阶段号。
        :return: 同阶段所有分支的机器。
        """
        result = []
        for ls in self.machine_nodes[stage_id]:
            result.extend(ls)
        return result

    def _build_machine_network(self):
        """
        建立机器（资源）网络。

        :return: 无。
        """
        node_id = 0
        # 添加节点
        for i in range(self.n_stages):
            stage_machine_nodes = []
            for j in range(self.stage_branches[i]):
                branch_machine_nodes = []
                for k in range(self.stage_branch_machines[i]):
                    self.machine_network.add_node(node_id, stage_id=i, branch_id=j, machine_id=k, status=0)
                    branch_machine_nodes.append(node_id)
                    node_id += 1
                stage_machine_nodes.append(branch_machine_nodes)
            self.machine_nodes.append(stage_machine_nodes)
        self.machine_network.add_node('S', stage_id=-1, branch_id=-1, machine_id=-1, status=0)
        self.machine_network.add_node('L', stage_id=-1, branch_id=-1, machine_id=-1, status=0)
        # 对不涉及AGV的节点添加卸船下游节点信息
        i = 0
        for j in range(self.stage_branches[i]):
            for k in range(self.stage_branch_machines[i]):
                self.machine_network.nodes[self.machine_nodes[i][j][k]]['downstream_nodes_disc'] = \
                    self.machine_nodes[i + 1][j][:int(round(self.stage_branch_machines[i + 1] / 2))]
        for i in [1, 4]:
            for j in range(self.stage_branches[i]):
                for k in range(self.stage_branch_machines[i]):
                    if k >= round(self.stage_branch_machines[i] / 2):  # 如果是后半部分机器，不添加下游卸船机器
                        d_nodes = []
                    else:
                        d_nodes = self.machine_nodes[i + 1][j]
                    self.machine_network.nodes[self.machine_nodes[i][j][k]]['downstream_nodes_disc'] = \
                        d_nodes
        # 对涉及AGV的节点添加卸船下游节点信息
        for i in [2, 3]:
            for j in range(self.stage_branches[i]):
                for k in range(self.stage_branch_machines[i]):
                    self.machine_network.nodes[self.machine_nodes[i][j][k]]['downstream_nodes_disc'] = \
                        self._get_all_branch_machines(i + 1)
        # 对ASC的节点添加卸船下游节点信息：L
        for i in [5]:
            for j in range(self.stage_branches[i]):
                for k in range(self.stage_branch_machines[i]):
                    self.machine_network.nodes[self.machine_nodes[i][j][k]]['downstream_nodes_disc'] = ['L']
        # 对不涉及AGV的节点添加装船下游节点信息
        for i in [1]:
            for j in range(self.stage_branches[i]):
                for k in range(self.stage_branch_machines[i]):
                    if k < round(self.stage_branch_machines[i] / 2):  # 如果是前半部分机器，不添加下游装船机器
                        d_nodes = []
                    else:
                        d_nodes = self.machine_nodes[i - 1][j]
                    self.machine_network.nodes[self.machine_nodes[i][j][k]]['downstream_nodes_load'] = \
                        d_nodes
        for i in [2, 5]:
            for j in range(self.stage_branches[i]):
                for k in range(self.stage_branch_machines[i]):
                    self.machine_network.nodes[self.machine_nodes[i][j][k]]['downstream_nodes_load'] = \
                        self.machine_nodes[i - 1][j][int(round(self.stage_branch_machines[i - 1] / 2)):]
        # 对涉及AGV的节点添加装船下游节点信息
        for i in [3, 4]:
            for j in range(self.stage_branches[i]):
                for k in range(self.stage_branch_machines[i]):
                    self.machine_network.nodes[self.machine_nodes[i][j][k]]['downstream_nodes_load'] = \
                        self._get_all_branch_machines(i - 1)
        # 对MT的节点添加装船下游节点信息:S
        for i in [0]:
            for j in range(self.stage_branches[i]):
                for k in range(self.stage_branch_machines[i]):
                    self.machine_network.nodes[self.machine_nodes[i][j][k]]['downstream_nodes_load'] = ['S']

    def _generate_task_network_node_pos(self):
        """
        生成任务网络节点的位置。

        :return: 每一个节点的位置，其中x轴和y轴位置的范围均为（0， 1）
        """
        pos = {}
        no_machines = sum([i * j for i, j in zip(self.stage_branches, self.stage_branch_machines)])
        horizontal_unit = 1 / (no_machines + 1)  # 横向节点之间的间距
        job_unit = 1 / self.n_jobs  # 不同任务之间的差距
        branch_unit = [job_unit / x for x in self.stage_branches]  # 不同分支之间的差距
        machine_unit = [branch_unit[i] / (self.stage_branch_machines[i] + 1) for i in
                        range(len(self.stage_branch_machines))]  # 机器
        for i in range(self.n_jobs):
            for j in range(self.n_stages):
                for k in range(self.stage_branches[j]):
                    for h in range(self.stage_branch_machines[j]):
                        if len(self.task_nodes[i][j][k]) > 0:
                            pos[self.task_nodes[i][j][k][h]] = (
                                horizontal_unit * (self.machine_nodes[j][k][h] + 1),
                                1 - (job_unit * i + branch_unit[j] * k + machine_unit[j] * (h + 1)))
        return pos

    def draw_task_network(self, _ax=None, _save_fig_path=None):
        """
        画任务网络图。

        :param _ax: 绘图区
        :param _save_fig_path: 图片保存路径和扩展名
        :return:无
        """
        if _ax is None:
            plt.figure(dpi=300)
            _ax = plt.gca()
        node_pos = self._generate_task_network_node_pos()
        nx.draw(self.task_network, pos=node_pos, ax=_ax, with_labels=True,
                font_color='white')
        plt.tight_layout()
        if _save_fig_path is not None:
            plt.savefig(_save_fig_path)
        plt.show()

    def _get_phase_branch_machine_product(self):
        """
        生成一个phase_id, branch_id, machine_id的交叉组合。

        :return:一个包含所有交叉组合的列表。
        """
        result = []
        for i in range(self.n_stages):
            for j in range(self.stage_branches[i]):
                for k in range(self.stage_branch_machines[i]):
                    result.append((i, j, k))
        return result

    def _generate_machine_network_node_pos(self):
        """
        生成机器网络节点的位置。

        :return: 每一个节点的位置，其中x轴和y轴未知的范围均为（0， 1）
        """
        pos = {}
        horizontal_unit = 1 / (self.n_stages + 2 + 1)
        branch_unit = [1 / x for x in self.stage_branches]  # 不同分支之间的差距
        machine_unit = [branch_unit[i] / (self.stage_branch_machines[i] + 1) for i in range(self.n_stages)]  # 机器
        for i, j, k in self._get_phase_branch_machine_product():
            node_id = self.machine_nodes[i][j][k]
            x_pos = (i + 1 + 1) * horizontal_unit
            y_pos = j * branch_unit[i] + (k + 1) * machine_unit[i]
            pos[node_id] = (x_pos, y_pos)
        pos['S'] = (horizontal_unit, 0.5)
        pos['L'] = (1 - horizontal_unit, 0.5)
        return pos

    def draw_machine_network(self, ax=None, status_color=('#2ca02c', '#1f77b4', '#ff7f0e')):
        """
        画机器网络图。

        :param ax: 绘图区。
        :param status_color: 空闲、卸货和装货状态下的颜色
        :return:无
        """
        text_properties = {'fontsize': 11, 'ha': 'center', 'va': 'center'}
        ax_is_none = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(15, 12))
            ax_is_none = True
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.05, 1.05)
        node_poss = self._generate_machine_network_node_pos()
        node_color = []
        for x in self.machine_network.nodes:
            node_color.append(status_color[self.machine_network.nodes[x]['status']])
        node_labels = {i: '' for i in range(len(self.machine_network) - 2)}
        # for job_id, job_attr in self.job_info.items():
        #     if job_attr['status'] == 'working':
        #         node_labels[
        #             self.machine_nodes[job_attr['phase_id']][job_attr['branch_id']][job_attr['machine_id']]] = job_id
        nx.draw(self.machine_network, pos=node_poss, ax=ax, with_labels=True,
                font_color='white', node_color=node_color, labels=node_labels)
        ax.text(node_poss['S'][0], 1, 'Ship', **text_properties)
        ax.text(node_poss['L'][0], 1, 'Yard', **text_properties)
        for i, label in enumerate(self.stages_name):
            ax.text(node_poss[self.machine_nodes[i][0][0]][0], 1, label, **text_properties)
        # disc_node_status = [self.job_info[i]['status'] for i in range(self.n_job) if self.job_info[i]['direction'] == 1]
        # load_node_status = [self.job_info[i]['status'] for i in range(self.n_job) if
        #                     self.job_info[i]['direction'] == -1]
        # ax.text(node_poss['S'][0], 0.60, 'Total:{}'.format(self.n_job_disc), **text_properties)
        # ax.text(node_poss['S'][0], 0.55, 'Remain:{}'.format(disc_node_status.count('waiting')), **text_properties)
        # ax.text(node_poss['S'][0], 0.45, 'Finish:{}'.format(disc_node_status.count('finishing')), **text_properties)
        # ax.text(node_poss['L'][0], 0.60, 'Total:{}'.format(self.n_job_load), **text_properties)
        # ax.text(node_poss['L'][0], 0.55, 'Remain:{}'.format(load_node_status.count('waiting')), **text_properties)
        # ax.text(node_poss['L'][0], 0.45, 'Finish:{}'.format(load_node_status.count('finishing')), **text_properties)
        for i, direction in zip([0, 1, -1], ['Idle', 'Disc', 'Load']):
            ax.plot(node_poss['L'][0] * 0.97, 0.15 + i * 0.05, 'o', markersize=10, color=status_color[i])
            ax.text(node_poss['L'][0], 0.15 + i * 0.05, direction, **text_properties)
        if ax_is_none:
            plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)
            plt.show()


class ACTSchedulingNetworkSolver(ACTSchedulingNetwork):
    def __init__(self, **_instance_setting):
        super(ACTSchedulingNetworkSolver, self).__init__(**_instance_setting)
        self.instance_setting = _instance_setting
        self.working_nodes = []
        self.finishing_nodes = []
        self.target_nodes = []
        self.task_provider = {}
        self.task_assigned = {}
        self.machine_status = []  # self.machine_status[phase_id][branch_id][machine_id] -> status <-[-1, 0, 1]
        self.machine_jobs = []
        self.clock = 0
        self.scheduling_records = None
        self.network_changing = []
        self._further_init_scheduling()

    def _further_init_scheduling(self):
        """
        初始化调度过程相关的记录。主要是为调度网络添加一些与调度相关的节点属性，并找出开始进行调度求解的点。

        :return: 无
        """
        self.scheduling_records = {x: {} for x in self.task_network.nodes}
        for i in range(self.n_stages):
            self.machine_status.append(
                [[0 for _ in range(self.stage_branch_machines[i])] for _ in range(self.stage_branches[i])])
        for i in range(self.n_stages):
            self.machine_jobs.append(
                [[[] for _ in range(self.stage_branch_machines[i])] for _ in range(self.stage_branches[i])])
        initial_nodes = []
        for node in self.task_network.nodes:
            direction = self.task_network.nodes[node]['direction']
            stage_id = self.task_network.nodes[node]['stage_id']
            if direction == 1 and stage_id == 0 or direction == -1 and stage_id == 5:
                initial_nodes.append(node)
                self.task_provider[node] = None
        self.target_nodes = initial_nodes

    # %% 调度属性
    def _get_task_node_info(self, _task_node):
        job_id = self.task_network.nodes[_task_node]['job_id']
        stage_id = self.task_network.nodes[_task_node]['stage_id']
        branch_id = self.task_network.nodes[_task_node]['branch_id']
        machine_id = self.task_network.nodes[_task_node]['machine_id']
        return job_id, stage_id, branch_id, machine_id

    def _get_setup_time(self, _task_node):
        job_id, stage_id, branch_id, machine_id = self._get_task_node_info(_task_node)
        if len(self.machine_jobs[stage_id][branch_id][machine_id]) == 0:
            return 0
        else:
            return self.setup_time[self.machine_jobs[stage_id][branch_id][machine_id][-1]][job_id][stage_id]

    @staticmethod
    def _sp_random(_available_nodes):
        return [random.random() for _ in _available_nodes]

    def _sp_degree(self, _available_nodes):
        return [len(self.task_network[x]) for x in _available_nodes]

    def _sp_processing_time(self, _available_nodes):
        return [self.task_network.nodes[x]['processing_time'] for x in _available_nodes]

    def _sp_eoc(self, _available_nodes):
        return [self.task_network.nodes[x]['processing_time'] + self._get_setup_time(x) for x in _available_nodes]

    def _sp_setup_time(self, _available_nodes):
        return [self._get_setup_time(x) for x in _available_nodes]

    def _sp_trpom(self, available_nodes):
        trpoms = []
        for available_node in available_nodes:
            current_node = available_node
            trpom = 0
            while True:
                downstream_nodes = []
                downstream_nodes_processing_time = []
                for adj_node in self.task_network[current_node]:
                    if self.task_network[current_node][adj_node]['type'] == 'Task Order':
                        downstream_nodes.append(adj_node)
                        downstream_nodes_processing_time.append(
                            self.task_network.nodes[adj_node]['processing_time'])
                if len(downstream_nodes) > 0:
                    current_node = downstream_nodes[
                        downstream_nodes_processing_time.index(max(downstream_nodes_processing_time))]
                    trpom += max(downstream_nodes_processing_time)
                else:
                    break
            trpoms.append(trpom)
        return trpoms

    def _sp_trpao(self, available_nodes):
        trpaos = []
        for available_node in available_nodes:
            trpaos.append(
                sum([self.task_network.nodes[x]['processing_time'] for x in self.task_network[available_node]]))
        return trpaos

    def _get_available_nodes_index(self, available_nodes):
        return [list(self.task_network.nodes).index(x) for x in available_nodes]

    def _sp_katz_centrality(self, available_nodes):
        n = 4
        s = 0.3  # 小于邻接矩阵的最大特征值的导数
        adj_matrix = nx.to_numpy_array(self.task_network)
        kz_matrix = adj_matrix * s
        temp_matrix = adj_matrix * s
        for _ in range(n - 1):
            temp_matrix = temp_matrix @ adj_matrix * s
            kz_matrix += temp_matrix
        kz_matrix = kz_matrix - np.diag(np.diag(kz_matrix))
        available_nodes_index = self._get_available_nodes_index(available_nodes)
        return [sum(kz_matrix[i]) for i in available_nodes_index]

    def _sp_semi_local_centrality(self, available_nodes, weight=False):
        all_semi_local_centrality = []
        n = 4
        adj_matrix = nx.to_numpy_array(self.task_network)
        semi_local_matrix = adj_matrix
        temp_matrix = adj_matrix
        for _ in range(n - 1):
            temp_matrix = temp_matrix @ adj_matrix
            semi_local_matrix += temp_matrix
        semi_local_nodes = (semi_local_matrix - np.diag(np.diag(semi_local_matrix))) > 0
        semi_local_degree = np.sum(semi_local_nodes, axis=1)
        for available_node in available_nodes:
            index = list(self.task_network.nodes).index(available_node)
            if weight:
                weight_semi_local_centrality = 0
                for i, node_flag in enumerate(semi_local_nodes[index]):
                    if node_flag:
                        weight_semi_local_centrality += \
                            self.task_network.nodes[list(self.task_network.nodes)[i]]['processing_time']
                all_semi_local_centrality.append(weight_semi_local_centrality)
            else:
                all_semi_local_centrality.append(semi_local_degree[index])
        return all_semi_local_centrality

    def _sp_wslc(self, available_nodes):
        return self._sp_semi_local_centrality(available_nodes, weight=True)

    def _sp_mapping_entropy(self, available_nodes):
        all_me = []
        all_nodes = list(self.task_network.nodes)
        all_node_degree = nx.degree(self.task_network)
        for available_node in available_nodes:
            adj_nodes = [x for x in self.task_network[available_node]]
            all_adj_dc = [np.log(all_node_degree[x] / (len(all_nodes) - 1)) for x in adj_nodes]
            all_me.append(all_node_degree[available_node] / (len(all_nodes) - 1) * sum(all_adj_dc))
        return all_me

    def _sp_k_shell(self, available_nodes):
        k_cores = nx.core_number(self.task_network)
        k_shells = [k_cores[x] for x in available_nodes]
        return k_shells

    def _sp_subgraph_centrality(self, available_nodes):
        p = 5
        sc = [0 for _ in available_nodes]
        available_nodes_index = self._get_available_nodes_index(available_nodes)
        adj_matrix = nx.to_numpy_array(self.task_network)
        temp_matrix = adj_matrix
        temp_p = 1
        for i in range(1, p + 1):
            temp_p *= i
            for j in range(len(available_nodes)):
                sc[j] += temp_matrix[available_nodes_index[j]][available_nodes_index[j]] / temp_p
            temp_matrix = temp_matrix @ adj_matrix
        return sc

    def _sp_collective_influence(self, available_nodes):
        ci = []
        all_degrees = nx.degree(self.task_network)
        for available_node in available_nodes:
            ci.append((all_degrees[available_node] - 1) * sum(
                [all_degrees[x] - 1 for x in self.task_network[available_node]]))
        return ci

    def _sp_clustering_coefficient(self, available_nodes):
        adj_matrix = nx.to_numpy_array(self.task_network)
        cc_matrix = adj_matrix @ adj_matrix @ adj_matrix
        available_nodes_index = self._get_available_nodes_index(available_nodes)
        all_degrees = nx.degree(self.task_network)
        cc = [cc_matrix[available_nodes_index[i]][available_nodes_index[i]] / (
                all_degrees[node] * (all_degrees[node] - 1) + 0.01) for i, node in enumerate(available_nodes)]
        return cc

    def _sp_closeness_centrality(self, available_nodes):
        close_values = nx.closeness_centrality(self.task_network)
        cloc_values = [close_values[x] for x in available_nodes]
        return cloc_values

    def _sp_betweenness_centrality(self, available_nodes):
        betweenness_values = nx.betweenness_centrality(self.task_network)
        bc_values = [betweenness_values[x] for x in available_nodes]
        return bc_values

    def _sp_eccentricity_centrality(self, available_nodes):
        eccentricity_values = nx.algorithms.distance_measures.eccentricity(self.task_network)
        ecc_values = [eccentricity_values[x] for x in available_nodes]
        return ecc_values

    def _sp_redundancy_centrality(self, available_nodes):
        sats_nodes = [x for x in available_nodes if len(self.task_network[x]) > 1]
        redundancy_values = nx.algorithms.bipartite.redundancy.node_redundancy(self.task_network, sats_nodes)
        redc_values = [-redundancy_values[x] if x in sats_nodes else -1 for x in available_nodes]
        return redc_values

    def _sp_structural_holes(self, available_nodes):
        sh = []
        for node in available_nodes:
            node_sh = 0
            node_degree = len(self.task_network[node])
            neigh_set = [x for x in self.task_network[node]]
            for neigh in neigh_set:
                neigh_sh = 1 / node_degree
                for another_neigh in neigh_set:
                    if neigh == another_neigh:
                        continue
                    if neigh in self.task_network[another_neigh]:
                        neigh_sh += 1 / node_degree * 1 / len(self.task_network[another_neigh])
                node_sh += neigh_sh ** 2
            sh.append(-node_sh)
        return sh

    def _sp_neighborhood_similarity(self, available_nodes):
        ns = []
        for node in available_nodes:
            node_ns = 0
            neigh_set = [x for x in self.task_network[node]]
            for neigh in neigh_set:
                for another_neigh in neigh_set:
                    if neigh == another_neigh:
                        continue
                    if neigh not in self.task_network[another_neigh] and another_neigh not in self.task_network[
                        neigh]:
                        neigh_neigh = set([x for x in self.task_network[neigh]])
                        another_neigh_neigh = set([x for x in self.task_network[another_neigh]])
                        node_ns += 1 - len(neigh_neigh & another_neigh_neigh) / len(neigh_neigh | another_neigh_neigh)
            ns.append(node_ns)
        return ns

    def _sp_resilience_centrality(self, available_nodes):
        # 下面的公式参见https://journals.aps.org/pre/abstract/10.1103/PhysRevE.101.022304#fulltext eq12
        resc = []
        in_degrees = self.task_network.in_degree()
        in_degrees = np.array([in_degrees[x] for x in self.task_network.nodes]).reshape(-1, 1)
        out_degrees = self.task_network.out_degree()
        out_degrees = np.array([out_degrees[x] for x in self.task_network.nodes]).reshape(-1, 1)
        adj_matrix = nx.to_numpy_array(self.task_network)
        for node in available_nodes:
            node_index = list(self.task_network.nodes).index(node)
            node_r = (((adj_matrix[node_index].reshape(1, -1) @ in_degrees +
                        adj_matrix[:, node_index].reshape(1, -1) @ out_degrees)[0][0] +
                       in_degrees[node_index] * out_degrees[node_index]) / (
                              (in_degrees.reshape(1, -1) @ out_degrees)[0][0] + 0.1)
                      -
                      (in_degrees[node_index] + out_degrees[node_index]) / (np.sum(in_degrees) + 0.1))
            resc.append(node_r)
        return resc

    def _transfer_node_id(self, _node, _job_id=None):
        """
        查找另外一个网络中当前节点的对应节点。如果提供job_id则查询task_node_id对应的machine_node_id，否则反过来。

        :param _node: 当前网络中节点。
        :param _job_id: 任务号。
        :return: 另一个网络中对应的节点id。
        """
        if _job_id is None:
            network = self.task_network
            nodes = self.machine_nodes
        else:
            network = self.machine_network
            nodes = self.task_nodes
        stage_id = network.nodes[_node]['stage_id']
        branch_id = network.nodes[_node]['branch_id']
        machine_id = network.nodes[_node]['machine_id']
        return nodes[stage_id][branch_id][machine_id] if _job_id is None else nodes[_job_id][stage_id][branch_id][
            machine_id]

    def _check_able_to_pick(self, _task_node):
        """
        首先通过agv任务的合理性校验，目前AGV不允许全部同向。
        然后判定MT,PT和ASC的关键任务节点，在下一步中转平台没有空闲的情况下，不允许抓箱子。

        :param _task_node: 任务节点。
        :return: bool。True代表可以被选中，False代表不可以被选中。
        """
        direction = self.task_network.nodes[_task_node]['direction']
        stage_id = self.task_network.nodes[_task_node]['stage_id']

        # %% 不允许同向左
        if stage_id == 3 and direction == -1:
            # AGV方向限制，不允许所有的agv同向左
            agv_status = self.machine_status[3][0]
            if len(set(agv_status + [direction])) == 2 and (agv_status + [direction]).count(0) == 1:
                return False

        # %% 对于关键位置：主小车和轨道吊，只要发现其对应任务的中转平台没有空，就不准开展任务
        downstream_nodes_label = 'downstream_nodes_disc' if direction == 1 else 'downstream_nodes_load'
        machine_node = self._transfer_node_id(_task_node)
        downstream_nodes = self.machine_network.nodes[machine_node][downstream_nodes_label]

        if (stage_id == 0 and direction == 1) or (stage_id == 5 and direction == -1):
            downstream_node_status = [self.machine_network.nodes[x]['status'] for x in downstream_nodes]
            if 0 not in downstream_node_status:
                return False

        return True

    def _check_no_active_neighbor(self, _node):
        """
        检查是否有邻居正在工作，如有则表明该节点不可被选用。

        :param _node:需要检查的节点
        :return:是否没有正在工作的邻居
        """
        for _node in self.task_network[_node]:
            if _node in self.working_nodes or _node in self.finishing_nodes:
                return False
        return True

    def _find_available_nodes(self):
        available_nodes = []
        for node in self.target_nodes:
            if self._check_able_to_pick(node) and self._check_no_active_neighbor(node):
                available_nodes.append(node)
        return available_nodes

    def _update_machine_status(self, _task_node, _direction=None):
        if _direction is None:
            _direction = self.task_network.nodes[_task_node]['direction']
        job_id, phase_id, branch_id, machine_id = self._get_task_node_info(_task_node)

        # # 更新任务进度信息
        # self.update_job_info(_task_node, direction)

        # # 记录下执行的任务序列
        # if direction in [-1, 1]:
        #     self.machines_jobs[phase_id][branch_id][machine_id].append(job_id)

        # 更新机器状态记录和机器网
        machine_node = self.machine_nodes[phase_id][branch_id][machine_id]
        self.machine_status[phase_id][branch_id][machine_id] = _direction
        self.machine_network.nodes[machine_node]['status'] = _direction
        d_nodes = copy.deepcopy(self.machine_network[machine_node].keys())
        # 理论上来说只有机器状态变为0才会有删除连边的需求，但是在研究过程中，可能会存在机器状态直接从1变为-1的情况，因此这里统一判定，
        # 只要机器状态发生改变，就删除原有连边。
        for d_node in d_nodes:
            self.machine_network.remove_edge(machine_node, d_node)
        if _direction in [-1, 1]:
            downstream_task_nodes = [d_node for d_node in self.task_network[_task_node] if
                                     self.task_network[_task_node][d_node]['type'] == 'Task Order']
            for d_t_node in downstream_task_nodes:
                _, d_stage_id, d_branch_id, d_machine_id = self._get_task_node_info(d_t_node)
                d_node = self.machine_nodes[d_stage_id][d_branch_id][d_machine_id]
                self.machine_network.add_edge(machine_node, d_node)
            # 对最后阶段执行的任务，添加下游节点信息。
            if phase_id == 5 and _direction == 1:
                self.machine_network.add_edge(machine_node, 'L')
            elif phase_id == 0 and _direction == -1:
                self.machine_network.add_edge(machine_node, 'S')

    def _remove_finishing_node(self, _task_node, ending_time):
        if _task_node is not None:
            self._update_machine_status(_task_node, 0)
            self.task_network.remove_node(_task_node)
            self.finishing_nodes.remove(_task_node)
            self.scheduling_records[_task_node]['ending_time'] = ending_time
            node_degrees = [value for key, value in nx.degree(self.task_network)]
            current_network_status = sum(node_degrees) / len(node_degrees) if len(
                node_degrees) > 1 else 0
            self.network_changing.append(current_network_status)

    def _get_last_job_ending_time(self, _task_node):
        job_id, stage_id, branch_id, machine_id = self._get_task_node_info(_task_node)
        preceding_jobs = self.machine_jobs[stage_id][branch_id][machine_id]
        if len(preceding_jobs) > 0:
            last_job = preceding_jobs[-1]
            return self.scheduling_records[self.task_nodes[last_job][stage_id][branch_id][machine_id]][
                'ending_time']
        else:
            return 0

    def _make_a_decision(self, _available_nodes, _scheduling_property, _minmax):
        # %% 决策
        _scheduling_property = eval('self._sp_{}'.format(_scheduling_property))
        _minmax = eval(_minmax)
        node_properties = _scheduling_property(_available_nodes)
        decision = _available_nodes[node_properties.index(_minmax(node_properties))]
        return decision

    def _target_2_working(self, decision):
        """返回决策的加工时间"""
        # %% 更新决策节点的信息
        self.working_nodes.append(decision)
        self.target_nodes.remove(decision)
        last_job_ending_time = self._get_last_job_ending_time(decision)
        setup_time = self._get_setup_time(decision)
        # 可以预先调度，但是如果机器资源刚刚释放，需要再等一下
        if last_job_ending_time + setup_time < self.clock:
            initialing_time = self.clock - setup_time
        else:
            initialing_time = last_job_ending_time
        self.scheduling_records[decision]['initialing_time'] = initialing_time
        self.scheduling_records[decision]['working_time'] = self.scheduling_records[decision][
                                                                'initialing_time'] + setup_time
        self.scheduling_records[decision]['finishing_time'] = self.scheduling_records[decision]['working_time'] + \
                                                              self.task_network.nodes[decision]['processing_time']
        job_id, stage_id, branch_id, machine_id = self._get_task_node_info(decision)
        self.machine_jobs[stage_id][branch_id][machine_id].append(job_id)
        self._update_machine_status(decision)
        for exclude_node in self.task_network.nodes[decision]['exclude_nodes']:
            if exclude_node in self.task_network.nodes:
                self.task_network.remove_node(exclude_node)
            if exclude_node in self.target_nodes:
                self.target_nodes.remove(exclude_node)
        # %% 更新决策节点父节点信息
        self._remove_finishing_node(self.task_provider[decision], self.scheduling_records[decision]['working_time'])
        return self.task_network.nodes[decision]['processing_time']

    def _working_2_finishing(self):
        if len(self.working_nodes) == 0:
            self.draw_machine_network()
            raise NameError('发生死锁！')
        else:
            # %% find next finishing node
            all_finishing_time = [self.scheduling_records[node]['finishing_time'] for node in self.working_nodes]
            self.clock = min(all_finishing_time)
            finishing_node = self.working_nodes[all_finishing_time.index(min(all_finishing_time))]
            self.finishing_nodes.append(finishing_node)
            self.working_nodes.remove(finishing_node)
            # %% add downstream nodes to target nodes
            downstream_nodes = []
            for adj_node in self.task_network[finishing_node]:
                if self.task_network[finishing_node][adj_node]['type'] == 'Task Order':
                    downstream_nodes.append(adj_node)
                    self.target_nodes.append(adj_node)
                    self.task_provider[adj_node] = finishing_node
            if len(downstream_nodes) == 0:
                self._remove_finishing_node(finishing_node, self.clock)

    def reset(self):
        """重置调度问题"""
        self.__init__(**self.instance_setting)
        available_nodes = self._find_available_nodes()
        return available_nodes

    def scheduling(self, _scheduling_property='random', _minmax='min', _random_seed=None):
        self.reset()
        random.seed(_random_seed)
        while len(self.task_network.nodes) > 0:
            available_nodes = self._find_available_nodes()
            while len(available_nodes) > 0:
                # sys.stdout.write('\r{:.20%}'.format(1 - len(self.task_network.nodes) / total_len))
                # sys.stdout.flush()
                decision = self._make_a_decision(available_nodes, _scheduling_property, _minmax)
                self._target_2_working(decision)
                available_nodes = self._find_available_nodes()
            self._working_2_finishing()
        random.seed(None)
        through_put = self.n_jobs / self.clock * 3600 / self.n_mts
        self.task_network = self.task_network_copy  # 还原网络
        return {'tp': round(through_put, 2),
                'scheduling_records': self.scheduling_records,
                'makespan': round(self.clock, 2),
                'network_changing': self.network_changing}

    def cal_state(self, available_nodes, attrs):
        # print('available nodes:', available_nodes)
        if len(available_nodes) == 0:  # 最终的状态
            return np.ones((len(attrs), len(attrs))).ravel()
        elif len(available_nodes) == 1:
            return np.zeros((len(attrs), len(attrs))).ravel()

        nodes_attrs = []
        for attr in attrs:
            fun = eval('self._sp_{}'.format(attr))
            nodes_attrs.append(fun(available_nodes))
        nodes_attrs = np.array(nodes_attrs)
        nodes_attrs += 0.01

        state = np.zeros((len(attrs), len(attrs)))
        for i in range(len(attrs)):
            for j in range(len(attrs)):
                state[i][j] = np.dot(nodes_attrs[i, :], nodes_attrs[j, :]) / np.linalg.norm(
                    nodes_attrs[i, :]) / np.linalg.norm(nodes_attrs[j, :])
        # print(state.ravel())
        return state.ravel()

    def step(self, action):
        """单步调度"""
        _minmax, _scheduling_property = action.split('-')
        # 先是根据当前的状态，进行决策，当前的状态下，available nodes一定不为空
        available_nodes = self._find_available_nodes()
        decision = self._make_a_decision(available_nodes, _scheduling_property, _minmax)
        reward = self._target_2_working(decision)
        # 寻找下一个状态，下一个状态是指下一个available nodes可用的时候
        while True:
            available_nodes = self._find_available_nodes()
            if len(available_nodes) > 0:
                done = False
                return available_nodes, 0, done
            else:
                self._working_2_finishing()
                if len(self.task_network.nodes) == 0:
                    done = True
                    self.task_network = self.task_network_copy
                    return [], 10000/self.clock, done

    def draw_gantt_chart(self, save_path=None):
        gantt_chart_info = {'S{}-{}'.format(i + 1, j + 1): [] for i in range(self.n_stages) for j in
                            range(self.stage_branches[i] * self.stage_branch_machines[i])}
        for key, value in self.scheduling_records.items():
            if 'initialing_time' in value.keys():
                job_id, stage_id, branch_id, machine_id = self._get_task_node_info(key)
                gantt_chart_info['S{}-{}'.format(stage_id + 1, branch_id * self.stage_branch_machines[
                    stage_id] + machine_id + 1)].append((job_id + 1,
                                                         value['initialing_time'],
                                                         value['working_time'],
                                                         value['finishing_time'],
                                                         value['ending_time']))
        code_010_preparation.draw_gantt_chart(gantt_chart_info, save_path)

    def analyze_scheduling_result(self, makespan):
        # 阶段切换时间与阶段等待时间
        utilization_times = [0 for _ in range(self.n_stages)]
        setup_times = [0 for _ in range(self.n_stages)]
        unload_waiting_times = [0 for _ in range(self.n_stages)]
        load_waiting_times = [0 for _ in range(self.n_stages)]
        for key, values in self.scheduling_records.items():
            if 'initialing_time' in values.keys():
                stage_id = self.task_network.nodes[key]['stage_id']
                direction = self.task_network.nodes[key]['direction']
                utilization_times[stage_id] += values['ending_time'] - values['initialing_time']
                setup_times[stage_id] += values['working_time'] - values['initialing_time']
                if direction == 1:
                    unload_waiting_times[stage_id] += values['ending_time'] - values['finishing_time']
                else:
                    load_waiting_times[stage_id] += values['ending_time'] - values['finishing_time']
        utilization_rates = [round(x / (self.stage_branches[i] * self.stage_branch_machines[i]) / makespan, 2)
                             for i, x in enumerate(utilization_times)]
        setup_times = [round(x / self.n_jobs, 2) for x in setup_times]
        unload_waiting_times = [round(x / self.n_jobs_disc, 2) for x in unload_waiting_times]
        load_waiting_times = [round(x / self.n_jobs_load, 2) for x in load_waiting_times]
        # 阶段重进重出率
        stage_dual_rates = []
        for stage in self.machine_jobs:
            stage_dual_rate = []
            for branch in stage:
                for machine in branch:
                    for i in range(len(machine) - 1):
                        # i < self.n_jobs_disc 表示卸船
                        if (machine[i] < self.n_jobs_disc and machine[i + 1] < self.n_jobs_disc) or (
                                (machine[i] >= self.n_jobs_disc and machine[i + 1] >= self.n_jobs_disc)):
                            stage_dual_rate.append(0)
                        else:
                            stage_dual_rate.append(1)
            if len(stage_dual_rate) == 0:
                stage_dual_rates.append(0)
            else:
                stage_dual_rates.append(round(sum(stage_dual_rate) / len(stage_dual_rate), 2))
        return {'utilization_rates': utilization_rates,
                'setup_times': setup_times,
                'unload_waiting_times': unload_waiting_times,
                'load_waiting_times': load_waiting_times,
                'stage_dual_rates': stage_dual_rates}


if __name__ == '__main__':
    instance_setting = {'n_mts': 2, 'n_tps_per_mt': 2, 'n_agvs': 10, 'n_mates_per_asc': 4, 'n_ascs': 3,
                        'n_jobs_disc': 50, 'n_jobs_load': 50, 'job_mt': None, 'job_asc': None, 'random_seed': 2}
    act_solver = ACTSchedulingNetworkSolver(**instance_setting)
    # act_solver.draw_task_network()
    scheduling_results = act_solver.scheduling('degree', 'min')
    # act_solver.draw_gantt_chart()
    # result = act_solver.analyze_scheduling_result(scheduling_results['makespan'])

    # scheduling_results_max = act_solver.scheduling('neighborhood_similarity', max)
    # act_solver.draw_gantt_chart()
    # result = act_solver.analyze_scheduling_result(scheduling_results['makespan'])
    # plt.figure(dpi=300)
    # plt.plot(scheduling_results['network_changing'], label='min neighborhood_similarity')
    # plt.plot(scheduling_results_max['network_changing'], label='max neighborhood_similarity')
    # plt.legend()
    # plt.xlabel('Time')
    # plt.ylabel('Average Degree')
    # plt.title('neighborhood_similarity')
    # plt.show()
