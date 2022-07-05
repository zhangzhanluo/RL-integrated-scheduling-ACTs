"""
    Author: Zhanluo Zhang
    Author E-mail: zhangzhanluo@outlook.com
    Version: v1.0
    Created Date: 20210826
    Description: task generation
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


class Instance:
    def __init__(self, **_instance_setting):
        """
        模拟港口调度问题的一些关键信息，例如任务加工时间和切换时间。如果不给定任务主小车和轨道吊分配情况，这里也会对其进行模拟。
        task_setting = {'n_mt': 2, 'n_tp_per_mt': 2, 'n_agv': 4, 'n_mate_per_asc': 2, 'n_asc': 2,
                             'n_job_disc': 4, 'n_job_load': 4, 'job_mt': None, 'job_asc': None}
        """
        self.stages_name = ['MT', 'TP', 'PT', 'LAGV', 'AGVM', 'ASC']
        self.n_stages = len(self.stages_name)
        self.n_jobs_disc = _instance_setting['n_jobs_disc']
        self.n_jobs_load = _instance_setting['n_jobs_load']
        self.n_jobs = self.n_jobs_disc + self.n_jobs_load
        self.n_mts = _instance_setting['n_mts']
        self.n_tps_per_mt = _instance_setting['n_tps_per_mt']
        self.n_agvs = _instance_setting['n_agvs']
        self.n_ascs = _instance_setting['n_ascs']
        self.n_mates_per_asc = _instance_setting['n_mates_per_asc']
        self.job_mt = _instance_setting['job_mt'] if 'job_mt' in _instance_setting.keys() else None
        self.job_asc = _instance_setting['job_asc'] if 'job_asc' in _instance_setting.keys() else None
        low_cv = 0.1
        mid_cv = 0.15
        high_cv = 0.2
        self.processing_time_setting = {'MT': {'load mean': 87, 'disc mean': 62, 'cv': mid_cv},
                                        'PT': {'load mean': 85, 'disc mean': 46, 'cv': low_cv},
                                        'LAGV': {'load mean': 290, 'disc mean': 300, 'cv': high_cv},
                                        'ASC': {'load mean': 91, 'disc mean': 86, 'cv': low_cv}}
        self.setup_time_setting = {'MT': {'load to load mean': 47, 'disc to disc mean': 80,
                                          'load to disc mean': 75, 'disc to load mean': 22,
                                          'cv': mid_cv},
                                   'PT': {'load to load mean': 36, 'disc to disc mean': 75,
                                          'load to disc mean': 22, 'disc to load mean': 50,
                                          'cv': low_cv},
                                   'LAGV': {'load to load mean': 200, 'disc to disc mean': 200,
                                           'load to disc mean': 10, 'disc to load mean': 10,
                                           'cv': high_cv},
                                   'ASC': {'load to load mean': 70, 'disc to disc mean': 64,
                                           'load to disc mean': 40, 'disc to load mean': 61,
                                           'cv': low_cv}}
        self.random_seed = _instance_setting[
            'random_seed'] if 'random_seed' in _instance_setting.keys() else 0  # 默认不要随机任务配置
        self._further_init()

    def _simulate_job_assignment(self, _n_machines):
        """
        需要搬运的集装箱可能和主小车（MT）或者轨道吊（ASC）具有不同的地理位置关系，如未提供真实的任务和设备相对地理位置关系，则适用本函数模拟生成。
        模拟的思路是先平均分配任务到每个设备，对于不够分的任务，随机从机器中挑选（剩余的任务的分配时机器可能会重复）.
        如果不想重复，把random.choices函数改为random.sample函数即可

        :param _n_machines: MT或者ASC设备数
        :return: 每个机器选定的设备号, np.array: (n_job,)
        """
        job_assignment = []
        for i in range(_n_machines):
            job_assignment.append(np.ones(self.n_jobs // _n_machines) * i)
        job_assignment.append(np.random.choice(range(_n_machines), self.n_jobs % _n_machines))
        job_assignment = np.concatenate(job_assignment, axis=0)
        np.random.shuffle(job_assignment)
        return job_assignment

    def _calculate_stage_processing_time(self, stage_name):
        disc_size = (self.n_jobs_disc, 1)
        load_size = (self.n_jobs_load, 1)
        return np.concatenate([
            np.random.standard_normal(disc_size) * self.processing_time_setting[stage_name]['disc mean'] *
            self.processing_time_setting[stage_name]['cv'] + self.processing_time_setting[stage_name]['disc mean'],
            np.random.standard_normal(load_size) * self.processing_time_setting[stage_name]['load mean'] *
            self.processing_time_setting[stage_name]['cv'] + self.processing_time_setting[stage_name]['load mean']
        ], axis=0)

    def _simulate_processing_time(self):
        """
        模拟节点加工时间。

        :return: np.array.[n_jobs * n_phases]
        """

        raw_size = (self.n_jobs, 1)
        t_processing = np.concatenate([
            self._calculate_stage_processing_time('MT'),
            np.zeros(raw_size),
            self._calculate_stage_processing_time('PT'),
            self._calculate_stage_processing_time('LAGV'),
            np.zeros(raw_size),
            self._calculate_stage_processing_time('ASC')
        ], axis=1)
        if t_processing.min() < 0:
            raise NameError('负的加工时间')
        return np.around(t_processing, 2)

    def _simulate_setup_time(self):
        setup_time = np.zeros((self.n_jobs, self.n_jobs, self.n_stages))
        for i in range(self.n_jobs):
            for j in range(self.n_jobs):
                for k in range(self.n_stages):
                    if i < self.n_jobs_disc:
                        if j < self.n_jobs_disc:
                            case = 'disc to disc mean'
                        else:
                            case = 'disc to load mean'
                    else:
                        if j < self.n_jobs_disc:
                            case = 'load to disc mean'
                        else:
                            case = 'load to load mean'
                    if k in [1, 4] or i == j:
                        continue
                    else:
                        setup_time[i][j][k] = np.random.standard_normal() * \
                                              self.setup_time_setting[self.stages_name[k]][case] * \
                                              self.setup_time_setting[self.stages_name[k]]['cv'] + \
                                              self.setup_time_setting[self.stages_name[k]][case]
        if setup_time.min() < 0:
            raise NameError('负的切换时间')
        return np.around(setup_time, 2)

    def _further_init(self):
        np.random.seed(self.random_seed)
        if self.job_mt is None:
            self.job_mt = self._simulate_job_assignment(self.n_mts)
        if self.job_asc is None:
            self.job_asc = self._simulate_job_assignment(self.n_ascs)
        if isinstance(self.job_asc, list):
            self.job_asc = np.array(self.job_asc)
        self.job_mt = np.array(self.job_mt) if isinstance(self.job_mt, list) else self.job_mt
        self.processing_time = self._simulate_processing_time()
        self.setup_time = self._simulate_setup_time()
        np.random.seed(None)  # None会让后面的随机数重回随机状态


def draw_gantt_chart(gantt_chart_info, save_path=None):
    """
    draw gantt chart
    :param gantt_chart_info: {'machine_name':[(job_id, starting_time, working_time, finishing_time, ending_time),(...)]}
    :param save_path: 图片保存路径和类型。
    :return: 无
    """
    machines_name = list(gantt_chart_info.keys())
    machines_name.sort()

    with plt.style.context('00_Files/ijpr_fullwidth.mplstyle'):
        for machine_name in machines_name:
            for block in gantt_chart_info[machine_name]:
                y = machines_name.index(machine_name)
                plt.barh(y, left=block[1], width=block[2] - block[1], height=0.9, color='#70ad47',
                         edgecolor='#70ad47', alpha=0.2)
                plt.barh(y, left=block[2], width=block[3] - block[2], height=0.9, color='#70ad47',
                         edgecolor='#70ad47', alpha=0.6)
                plt.barh(y, left=block[3], width=block[4] - block[3], height=0.9, color='#70ad47',
                         edgecolor='#70ad47', alpha=1)
                plt.text((block[1] + block[4]) * 0.5, y, block[0],
                         ha='center', va='center', fontsize=8)
        plt.yticks(range(len(machines_name)), machines_name)
        plt.xlabel('Time')
        plt.ylabel('Machine')

        legend_y = -0.11
        legend_height = 0.035
        legend_width = 0.05
        plt.gcf().add_artist(Rectangle((0.64, legend_y), width=legend_width, height=legend_height,
                                       facecolor='#70ad47', edgecolor='#70ad47', alpha=0.2,
                                       transform=plt.gca().transAxes))
        plt.text(0.695, legend_y + legend_height / 2, 'setup', transform=plt.gca().transAxes, ha='left', va='center',
                 fontsize=plt.rcParams['xtick.labelsize'])
        plt.gcf().add_artist(Rectangle((0.76, legend_y), width=legend_width, height=legend_height,
                                       facecolor='#70ad47', edgecolor='#70ad47', alpha=0.6,
                                       transform=plt.gca().transAxes))
        plt.text(0.815, legend_y + legend_height / 2, 'work', transform=plt.gca().transAxes, ha='left', va='center',
                 fontsize=plt.rcParams['xtick.labelsize'])
        plt.gcf().add_artist(Rectangle((0.88, legend_y), width=legend_width, height=legend_height,
                                       facecolor='#70ad47', edgecolor='#70ad47', alpha=1,
                                       transform=plt.gca().transAxes))
        plt.text(0.935, legend_y + legend_height / 2, 'wait', transform=plt.gca().transAxes, ha='left', va='center',
                 fontsize=plt.rcParams['xtick.labelsize'])
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()


if __name__ == '__main__':
    instance_setting = {'n_mts': 2, 'n_tps_per_mt': 2, 'n_agvs': 12, 'n_mates_per_asc': 2, 'n_ascs': 3,
                        'n_jobs_disc': 100, 'n_jobs_load': 100, 'job_mt': None, 'job_asc': None}
    instance = Instance(**instance_setting)
