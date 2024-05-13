'''
author: lcy
This file provides the python interfaces of the routers simulations environment
this file do the interactions between ryu+mininet
'''
from gym import spaces
if __name__ == "__main__":
    from utils import weight_choice
else:
    from net_env.utils import weight_choice
import torch
import random
import numpy as np
import os
import sys
import glob
import heapq
import copy
import json
import socket
import argparse

class Request():
    def __init__(self, s, t, start_time, end_time, demand, rtype):
        # use open time interval: start_time <= time < end_time
        self.s = s
        self.t = t
        self.start_time = start_time
        self.end_time = end_time
        self.demand = demand
        self.rtype = rtype
    '''
        deprecated function
    '''
    def to_json(self):
        data = {
            'src': int(self.s),
            'dst': int(self.t),
            'time': int(self.end_time - self.start_time),
            'rtype': int(self.rtype),
            'demand': int(self.demand),
                }
        return json.dumps(data)

    
    def __lt__(self, other):
        return self.end_time < other.end_time
    
    def __str__(self):
        return "s: %d t: %d\nstart_time: %d\nend_time: %d\ndemand: %d\nrtype: %d" % (self.s, self.t, self.start_time, self.end_time, self.demand, self.rtype)

class NetEnv():
    '''
    must run setup before using other methods
    '''
    def __init__(self, args):
        self.args = args

        self._observation_spaces = []
        self._action_spaces = []
        self._delay_discounted_factor = 0.99
        self._loss_discounted_factor = 0.99

        # set up communication to remote hosts(mininet host)
        MININET_HOST_IP = '127.0.0.1'   # Testbed server IP
        MININET_HOST_PORT = 5000
        self.BUFFER_SIZE = 1024
        self.mininet_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.mininet_socket.connect((MININET_HOST_IP, MININET_HOST_PORT))
        
        CONTROLLER_IP = '127.0.0.1'
        CONTROLLER_PORT = 3999
        self.controller_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.controller_socket.connect((CONTROLLER_IP, CONTROLLER_PORT))
        
    
    @property
    def observation_spaces(self):
        return self._observation_spaces
    
    @property
    def action_spaces(self):
        return self._action_spaces

    '''
    @param:
        toponame: a string indicate the toponame for the environment
    @retval:
        node_num: an int
        observation_spaces: a [node_name] shape list shows the observation spaces for each node, now are Box(k)
        action_spaces: a [node_name] shape list shows the observation spaces for each node, now are Discrete(l)
    '''
    def setup(self, toponame, demand_matrix_name):
        # toponame = Abi
        # demand_matrix_name = Abi_500.txt
        self._time_step = 0
        self._request_heapq = []  # 请求的堆？

        # init DiffServ info
        # 初始化不同流的服务类型
        self._type_num = 4
        self._type_dist = np.array([0.2, 0.3, 0.3, 0.2])  # TO BE CHECKED BEFORE EXPERIMENT
        # 每个流不同类型的概率

        # load topo info(direct paragraph)
        if toponame == "test":
            self._node_num = 4
            self._edge_num = 8
            self._observation_spaces = []
            self._action_spaces = []
            # topology info
            self._link_lists = [[3, 1], [0, 2], [1, 3], [2, 0]]
            # 针对这4个点，每个点（下标i）对应的边的节点，例如第0个点连接第3个和第1个点
            self._shr_dist = [[0, 1, 2, 1], [1, 0, 1, 2], [2, 1, 0, 1], [1, 2, 1, 0]]
            self._link_capa = [[0, 1000, 0, 5000], [1000, 0, 5000, 0], [0, 5000, 0, 5000], [5000, 0, 5000, 0]] # link capacity (Kbps)
            self._link_usage = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]] # usage of link capacity
            self._link_losses = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]] # link losses, x% x is int
            # request generating setting 请求生成的设置
            # bandwidth aware usually need as large bandwidth as they can, delay-bandwidth aware may have several stage of bandwidth demand
            self._request_demands = [[100], [500], [500], [100]]   
            self._request_times = [[10], [10], [10], [10]]  # simple example test
            self._node_to_agent = [0, 1, 2, 3]  #  i:None means drl-agent not deploys on node i; i:j means drl-agent j deploys on node i
            self._agent_to_node = [0, 1, 2, 3]  # indicate the ith agent's node
            self._agent_num = 4
            self._demand_matrix = [0, 1, 1, 1,
                                   1, 0, 1, 1,
                                   1, 1, 0, 1,
                                   1, 1, 1, 0]

            for i in self._agent_to_node:
                # onehot src, onehot dst, neighbour shr dist to each node
                # low and high for Box isn't essential
                self._observation_spaces.append(spaces.Box(0., 1., [1 + self._node_num * len(self._link_lists[i]) + self._node_num * len(self._link_lists[i]) + self._node_num ** 2 + self._node_num ** 2 + self._type_num + self._node_num * 2], dtype=np.float32)) # maximum observation space
                #self._observation_spaces.append(spaces.Box(0., 1., [1 + len(self._link_lists[i]) + len(self._link_lists[i]) + self._type_num + self._node_num * 2], dtype=np.float32)) # minimum observation space
                # 0.和1.表示浮点型数据
                # 这里还是不明白，大概就是观察空间的min、max、维数和数据类型
                self._action_spaces.append(spaces.Discrete(2))
                # 动作空间只是离散的两个值
            
        elif toponame == "Abi":
            self._load_topology(toponame, demand_matrix_name)
        elif toponame == "GEA":
            self._load_topology(toponame, demand_matrix_name)
        else:
            raise NotImplementedError
        return self._agent_num, self._node_num, self._observation_spaces, self._action_spaces, self._type_num

    '''
    reset the environment
    @retval:
        states: [torch.tensor([x, y, ...]), ...]
    '''
    def reset(self):
        self._time_step = 0
        self._request_heapq = []
        self._link_usage = [([0.] * self._node_num) for i in range(self._node_num)]
        self._delay_normal = [([1.] * self._node_num) for i in range(self._node_num)]
        self._loss_normal = [([1.] * self._node_num) for i in range(self._node_num)]
        self._update_state()
        return self._request, self._states
    # 返回新生成的请求和环境各种状态
    
    '''
    interact with controller and mininet to install path rules and generate request
    与控制器和mininet交互，安装路径规则和生成请求
    @retval:
        metrics of path including throughput, packet loss 
        路径指标包括吞吐量，包丢失
    '''
    def sim_interact(self, request, path):
        # install path in controller
        data_js = {}
        data_js['path'] = path
        data_js['ipv4_src'] = "10.0.0.%d" % (request.s + 1)
        data_js['ipv4_dst'] = "10.0.0.%d" % (request.t + 1)
        data_js['src_port'] = self._time_step % 10000 + 10000
        data_js['dst_port'] = self._time_step % 10000 + 10000 
        msg = json.dumps(data_js)
        self.controller_socket.send(msg.encode())
        self.controller_socket.recv(self.BUFFER_SIZE)
        
        # communicate to testbed
        data_js = {}
        data_js['src'] = int(request.s)
        data_js['dst'] = int(request.t)
        data_js['src_port'] = int(self._time_step % 10000 + 10000)
        data_js['dst_port'] = int(self._time_step % 10000 + 10000) 
        data_js['rtype'] = int(request.rtype)
        data_js['demand'] = int(request.demand)
        data_js['rtime'] = int(request.end_time - request.start_time)
        msg = json.dumps(data_js)
        self.mininet_socket.send(msg.encode())
        # get the feedback
        msg = self.mininet_socket.recv(self.BUFFER_SIZE)
        data_js = json.loads(msg)
        return data_js
        


    '''
    use action to interact with the environment
    @param: 
        actions:[torch.tensor([x]), ... the next hop action for each agent
        gfactors: list:shape[node_num] shows the reward factor for the global and local rwd
    @retval:
        states: [torch.tensor([x, y, ...]), ...]
        rewards: [torch.tensor([r]), ...]
        path: [start, x, y, z, ..., end]
    '''
    def step(self, actions, gfactors, simenv=True):
        # update
        path = [self._request.s]
        count = 0
        capacity = 1e9
        pre_node = None
        circle_flag = 0
        link_flag = [[0] * self._node_num for i in range(self._node_num)]
        node_flag = [0] * self._node_num
        node_flag[self._request.s] = 1
        while(count < self._node_num):
            curr_node = path[count]
            agent_ind = self._node_to_agent[curr_node]
            if agent_ind != None:
                next_hop = self._link_lists[curr_node][actions[agent_ind][0].item()]
            else:
                temp = [None, 1e9]
                for i in self._link_lists[curr_node]:
                    if i != pre_node and self._shr_dist[i][self._request.t] < temp[1]:
                        temp = [i, self._shr_dist[i][self._request.t]]
                next_hop = temp[0]
            
            if link_flag[curr_node][next_hop] == 1:
                # 把所有循环都移除后，由于没有生成的策略，进入这个if里面
                circle_flag = 1
                path = self.calcSHR(self._request.s, self._request.t)
                count = len(path) - 1
                
                break
            else:
                link_flag[curr_node][next_hop] = 1
            # delete ring
            # main中生成的path是有循环的，这一步把产生循环下一条的节点移除path
            if node_flag[next_hop] == 1:
                while path[count] != next_hop:
                    node_flag[path[count]] = 0
                    path.pop()
                    count -= 1
            else:
                path.append(next_hop)
                node_flag[next_hop] = 1
                count += 1
            if next_hop == self._request.t:
                break
            pre_node = curr_node
        
        # extra safe learning part to congestion prevention
        
        # for link failure
        for i in range(len(path) - 1):
            if self._link_capa[path[i]][path[i + 1]] == 0:
                circle_flag = 1
                path = self.calcSHR(self._request.s, self._request.t)
                count = len(path) - 1
                break
            # 如果链路中断（capa=0）就用最短路径策略生成path
            # circle_flag不完全表示有循环，例如，链路失败等各种不安全的路径都会用这个指标标记
        # a looser fallback policy trigger
        
        threshold = 1.2  # 2 for heavy load
        for i in range(len(path) - 1):
            if self._link_usage[path[i]][path[i + 1]] > threshold * self._link_capa[path[i]][path[i + 1]]:
                # 使用严重超载情况
                circle_flag = 1  # indicating safe learning approach being used
                path = self.calcBCSHR(self._request.s, self._request.t, self._request.demand)
                if path == None:
                    path = self.calcWP(self._request.s, self._request.t)
                count = len(path) - 1
                break
        
        
        # update sim network state
        # 根据path更新state状态
        capacity = 1e9
        for i in range(len(path) - 1):
            capacity = min(capacity, max(0, self._link_capa[path[i]][path[i + 1]] - self._link_usage[path[i]][path[i + 1]]))
            # 目前这个路径上还剩的容量
            self._link_usage[path[i]][path[i + 1]] += self._request.demand
            # 这次req之后，这条路径使用的容量
        capacity = max(capacity, 0)
        # path路径上各个link中剩余容量最少的那个
        self._request.path = copy.copy(path)
        
        #interact with sim-env(ryu + mininet)
        if simenv: # 这里初始是false
            ret_data = self.sim_interact(self._request, path)
            
            # for reliability test
            if 'change' in ret_data:
                self.change_env(ret_data['change'])

            delay = ret_data['delay']
            throughput = ret_data['throughput']
            loss_rate = ret_data['loss']
            delay_cut = min(1000, delay) # since 1000 is much larger than common delay
            # 原论文的第4页
            self._delay_normal[self._request.s][self._request.t] = self._delay_discounted_factor * self._delay_normal[self._request.s][self._request.t] + (1 - self._delay_discounted_factor) * delay_cut
            # 原文的公式（4），_delay_normal表示与第t个流具有相同源和目标节点的最近流的平均性能度量值
            # _delay_discounted_factor控制_delay_normal的更新速度
            delay_scaled = delay_cut / self._delay_normal[self._request.s][self._request.t]
            # 原文的公式（3），标准化原始的延迟
            delay_sq = - delay_scaled ** 1
            # 原文的公式（2）中的延迟y部分

            # loss同上
            self._loss_normal[self._request.s][self._request.t] = self._loss_discounted_factor * self._loss_normal[self._request.s][self._request.t] + (1 - self._loss_discounted_factor) * loss_rate
            loss_scaled = loss_rate / (0.01 + self._loss_normal[self._request.s][self._request.t])
            # 为什么loss的标准要+0.01？？？
            loss_sq = - loss_scaled ** 1

            # 吞吐量也在P4上半部分
            throughput_log = np.log(0.5 + throughput / self._request.demand) #avoid nan
            # 为什么要加0.5？？？
        else:
            delay = 0.
            throughput = 0.
            loss_rate = 0.
            delay_scaled = 0.
            delay_sq = 0.
            loss_sq = 0.
            throughput_log = 0.
        
        # calc global rwd of the generated path 
        if self._request.rtype == 0:
            global_rwd = 1 * delay_sq
        elif self._request.rtype == 1:
            global_rwd =  0. * (delay_sq) + 1 * throughput_log - 0. * min(self._request.demand / (capacity + 1), 1)
        elif self._request.rtype == 2:
            global_rwd = 0.5 * delay_sq + 0.5 * throughput_log
        else:
            global_rwd = 0.5 * delay_sq + 0.5 * loss_sq
        # avoid unsafe route 
        # fall back policy penalty
        # 前面检测到有循环，这里就在计算的奖励中引入惩罚值
        if circle_flag == 1:
            global_rwd -= 5
        

        rewards = []
        for i in range(self._agent_num):
            # add a dist reward for each node
            # local rwd = delta dist for the action each node do 
            ind = self._agent_to_node[i]
            if ind == self._request.t:
                local_rwd = 0.
            else:
                action_hop = self._link_lists[ind][actions[i][0].item()]
                # 生成本地的局部奖励值（评估下一跳跟最短距离比怎么样）
                # 算法生成的下一跳到dist与当前节点到dist的差值/当前节点到dist的差值
                # 评估是不是绕远了
                local_rwd =  (self._shr_dist[ind][self._request.t] - self._shr_dist[action_hop][self._request.t] - 1) / self._shr_dist[ind][self._request.t]
            rewards.append(torch.tensor([0.1 * (gfactors[i] * global_rwd  + (1 - gfactors[i]) * local_rwd)]))
            # 预训练的阶段：gfactors全为0，就是不关注全局的奖励，只关注本地奖励，即优化最短距离route，0.1*local_rwd
            # 训练的阶段：gfactors全为1，就是不关注局部的奖励，只关注全局奖励，即实际环境，0.1*global_rwd

        
        delta_dist = count - self._shr_dist[self._request.s][self._request.t]  # 评估跳数，与最短距离算法比
        delta_demand = min(capacity, self._request.demand) / self._request.demand   # 评估剩余容量满足需求的程度
        throughput_rate = throughput / self._request.demand
        rtype = self._request.rtype
        

        # generate new state
        self._time_step += 1
        self._update_state()
        return self._states, rewards, path, delta_dist, delta_demand, circle_flag, rtype, global_rwd, delay, throughput_rate, loss_rate
    
    '''
    giving current agent and it's action, return next agent ind and the nodes in the path
    agent = None means that path termiatied before meet a agent(or agent on the t node)
    @retval
        agent: the index of next agent
        path: the path node from 
    '''
    def next_agent(self, agent, action):
        curr_node = self._agent_to_node[agent]
        path = [] # not include current agent's node
        pre_node = curr_node
        action_hop = self._link_lists[curr_node][action[0].item()]
        curr_node = action_hop 
        while(1):
            path.append(curr_node)
            if curr_node == self._request.t:
                return None, path
            if self._node_to_agent[curr_node] != None:
                break
            temp = [None, 1e9]
            for i in self._link_lists[curr_node]:
                if i != pre_node and self._shr_dist[i][self._request.t] < temp[1]:
                    temp = [i, self._shr_dist[i][self._request.t]]
            pre_node = curr_node
            curr_node = temp[0]
        return self._node_to_agent[path[-1]], path

    '''
    return first agent index and the nodes in the path
    @retval
        agent: the index of next agent
        path: the path node from 
    '''
    def first_agent(self):
        path = [self._request.s]
        pre_node = None
        while(1):
            curr_node = path[-1]
            if curr_node == self._request.t:
                return None, path
            if self._node_to_agent[curr_node] != None:
                break
            temp = [None, 1e9]
            for i in self._link_lists[curr_node]:
                if i != pre_node and self._shr_dist[i][self._request.t] < temp[1]:
                    temp = [i, self._shr_dist[i][self._request.t]]
            path.append(temp[0])
            pre_node = curr_node
        return self._node_to_agent[path[-1]], path


            

    '''
    generate requests and update the state of environment
    生成请求并更新环境状态
    '''
    def _update_state(self):
        # update env request heapq
        while len(self._request_heapq) > 0 and self._request_heapq[0].end_time <= self._time_step:
            # 这里的_request_heapq[0].end_time就是第一个请求的end_time,应该是50
            # 进入条件是：请求堆里一直有请求，且已经处理的请求>50个时
            request = heapq.heappop(self._request_heapq)
            path = request.path  # 这里的path要关注！！！
            if path != None:
                for i in range(len(path) - 1):
                    self._link_usage[path[i]][path[i + 1]] -= request.demand
            # 把请求对应的path上的每个节点“link使用”数组减去请求的需求速率
            # 这里就是说当处理的请求超过50个之后，就把以前的请求弹出，将链路的使用痕迹去掉，定时更新

        # generate new request
        nodelist = range(self._node_num)
        # [0,1,2,3]
        # uniform sampling
        # s, t = random.sample(nodelist, 2)
        # sampling according to demand matrix
        ind = weight_choice(self._demand_matrix)
        # 在net_env/utils里面。从Abi中随机选取一个权重并且返回这个权重的位置index
        s = ind // self._node_num  #  整除
        t = ind % self._node_num   #  取余
        
        start_time = self._time_step
        rtype = np.random.choice(list(range(self._type_num)), p=self._type_dist)
        # 从[0，1，2，3]中按照[0.2, 0.3, 0.3, 0.2]概率随机抽取流的类型
        demand = random.choice(self._request_demands[rtype])  # 从这个类型的流的需求速率中随机抽取作为速率需求
        end_time = start_time + random.choice(self._request_times[rtype])  # 从10中rand一个流的持续时间
        print("start_time:", start_time, "end_time:", end_time)

        self._request = Request(s, t, start_time, end_time, demand, rtype)  # 生成请求
        heapq.heappush(self._request_heapq, self._request)  # 新生成的请求压入请求堆里

        # calc wp dist for each node pair
        # 针对一对节点，_wp_dist存这个两个点到外面任意一点（包括对端）链路的最大剩余容量
        self._wp_dist = []
        for i in range(self._node_num):
            self._wp_dist.append([])
            for j in range(self._node_num):
                if j == i:
                    self._wp_dist[i].append(1e6)  # not aware of i -> i
                elif j in self._link_lists[i]:
                    self._wp_dist[i].append(self._link_capa[i][j] - self._link_usage[i][j])
                else:
                    self._wp_dist[i].append(- 1e6)  # testing for heavyload
        for k in range(self._node_num):
            for i in range(self._node_num):
                for j in range(self._node_num):
                    if self._wp_dist[i][j] < min(self._wp_dist[i][k], self._wp_dist[k][j]):
                        self._wp_dist[i][j] = min(self._wp_dist[i][k], self._wp_dist[k][j]) 

        # generate the output state of environment
        # common state for each agent
        link_usage_info = []
        for j in range(self._node_num):
            for k in range(self._node_num):
                link_usage_info.append(self._link_capa[j][k] - self._link_usage[j][k])
        # 现有每个link的剩余容量
        
        link_loss_info = []
        for j in range(self._node_num):
            for k in range(self._node_num):
                link_loss_info.append(self._link_losses[j][k] / 100)  # input linke loss x indicating x%
        # 现有每个link的损失率

        self._states = []
        for i in self._agent_to_node:
            # generate src and dst one hot state
            type_state = torch.tensor(list(np.eye(self._type_num)[self._request.rtype]))
            # eye(4)生成一个对角为1的矩阵：
            # [[1,0,0,0],
            #  [0,1,0,0],
            #  [0,0,1,0],
            #  [0,0,0,1]]
            # 后面eye(4)[rtype]表示第rtype行的数据，例第3个类型的type_state[0,0,1,0]
            src_state = torch.tensor(list(np.eye(self._node_num)[self._request.s]))
            dst_state = torch.tensor(list(np.eye(self._node_num)[self._request.t]))
            one_hot_state = torch.cat([type_state, src_state, dst_state], 0)  # tensor拼接到一起
            # 新生成的请求，生成其类型、源、目的节点的one-hot

            # generate neighbors shr distance state
            neighbor_dist_state = []
            for j in self._link_lists[i]:
                neighbor_dist_state += self._shr_dist[j]
                #neighbor_dist_state.append(self._shr_dist[j][self._request.t]) #for less state
            neighbor_dist_state = torch.tensor(neighbor_dist_state, dtype=torch.float32)
            # 例如对于节点0，这个[]为：[[第3个节点到各节点的最短距离],[第1个节点的shr-dist]]
            
            # generate link edge state
            link_usage_state = torch.tensor(link_usage_info, dtype=torch.float32)
            
            # generate link loss state
            link_loss_state = torch.tensor(link_loss_info, dtype=torch.float32)

            # generate demand and time state
            extra_info_state = torch.tensor([self._request.demand], dtype=torch.float32)

            # generate neighbors widest path state
            neighbor_wp_state = []
            for j in self._link_lists[i]:
                neighbor_wp_state += self._wp_dist[j]
                #neighbor_wp_state.append(self._wp_dist[j][self._request.t]) #for less state
            neighbor_wp_state = torch.tensor(neighbor_wp_state, dtype=torch.float32)
            
            concat_state = torch.cat([extra_info_state, neighbor_wp_state, neighbor_dist_state, link_usage_state, link_loss_state, one_hot_state], 0) # full state
            #concat_state = torch.cat([extra_info_state, neighbor_wp_state, neighbor_dist_state, one_hot_state], 0)  # less state
            self._states.append(concat_state)
    
    '''
    load the topo and setup the environment
    '''
    def _load_topology(self, toponame, demand_matrix_name):
        data_path = os.path.dirname(os.path.realpath(__file__)) + "/inputs/"
        topofile = open(data_path + toponame + "/" + toponame + ".txt", "r")
        demandfile = open(data_path + toponame + "/" + demand_matrix_name, "r")
        self._demand_matrix = list(map(int, demandfile.readline().split()))
        # 不太知道_demand_matrix具体是什么，这里就是读取了Abi_500
        # the input file is undirected graph while here we use directed graph
        # node id for input file indexed from 1 while here from 0
        self._node_num, edge_num = list(map(int, topofile.readline().split()))
        self._edge_num = edge_num * 2
        self._observation_spaces = []
        self._action_spaces = []
        
        # build the link list
        self._link_lists = [[] for i in range(self._node_num)] # neighbor for each node
        self._link_capa = [([0] * self._node_num) for i in range(self._node_num)]  # 同test
        self._link_usage = [([0] * self._node_num) for i in range(self._node_num)]
        self._link_losses = [([0] * self._node_num) for i in range(self._node_num)]
        for i in range(edge_num):
            u, v, _, c, loss = list(map(int, topofile.readline().split()))
            # since node index range from 1 to n in input file
            self._link_lists[u - 1].append(v - 1)
            self._link_lists[v - 1].append(u - 1)
            # undirected graph to directed graph
            self._link_capa[u - 1][v - 1] = c
            self._link_capa[v - 1][u - 1] = c
            self._link_losses[u - 1][v - 1] = loss
            self._link_losses[v - 1][u - 1] = loss
        
        # input agent index
        is_agent = list(map(int, topofile.readline().split()))  # 最后一行表示每个节点是否配置agent
        self._agent_to_node = []
        self._node_to_agent = [None] * self._node_num
        for i in range(self._node_num):
            if is_agent[i] == 1:
                self._node_to_agent[i] = len(self._agent_to_node)
                self._agent_to_node.append(i)
        self._agent_num = len(self._agent_to_node)

        # calsulate shortest path distance
        self._shr_dist = []
        for i in range(self._node_num):
            self._shr_dist.append([])
            for j in range(self._node_num):
                if j == i:
                    self._shr_dist[i].append(0)
                elif j in self._link_lists[i]:
                    self._shr_dist[i].append(1)
                else:
                    self._shr_dist[i].append(1e6)  # inf
        for k in range(self._node_num):
            for i in range(self._node_num):
                for j in range(self._node_num):
                    if(self._shr_dist[i][j] > self._shr_dist[i][k] + self._shr_dist[k][j]):
                        self._shr_dist[i][j] = self._shr_dist[i][k] + self._shr_dist[k][j] 
        
        # generate observation spaces and action spaces
        for i in self._agent_to_node:
            # state: extra_state + neighbor_wp + neighbor shr(or least delay) + linkusage + link_losses + onehot type state + onehot src + dst state
            #  上面是什么意思？？？？？？
            self._observation_spaces.append(spaces.Box(0., 1., [1 + self._node_num * len(self._link_lists[i]) + self._node_num * len(self._link_lists[i]) + self._node_num ** 2 + self._node_num ** 2 + self._type_num + self._node_num * 2], dtype=np.float32)) # maximum observation state
            #self._observation_spaces.append(spaces.Box(0., 1., [1 + len(self._link_lists[i]) + len(self._link_lists[i]) + self._type_num + self._node_num * 2], dtype=np.float32)) # only use neighbor state space
            self._action_spaces.append(spaces.Discrete(len(self._link_lists[i])))
        
        # TO BE CHECKED BEFORE EXPERIMENT
        # setup flow generating step
        if toponame == "Abi":
            self._request_demands = [[100], [1500], [1500], [500]]
            # 类型I、类型II、类型III和类型IV的流量发送速率固定为100kbps、1500kbps、1500kbps和500kbps。
            self._request_times = [[50], [50], [50], [50]] # heavy load
            # 重负载场景，其中流持续时间为50时隙
            #self._request_times = [[10], [10], [10], [10]] # light load
            # 轻负载场景，流持续时间为10时隙
            #self._request_times = [[30], [30], [30], [30]] # mid load
        elif toponame == "GEA":
            self._request_demands = [[100], [1500], [1500], [500]]
            self._request_times = [[15], [15], [15], [15]]
    
    '''
        changing network status:
        link failure
        demand changing
    '''
    def change_env(self, msg):
        if msg == "link_failure":
            # Abi link failure 1-5
            self._link_capa[0][4] = 0
            self._link_capa[4][0] = 0
            
            # calculate shortest path distance for new topology
            self._shr_dist = []
            for i in range(self._node_num):
                self._shr_dist.append([])
                for j in range(self._node_num):
                    if j == i:
                        self._shr_dist[i].append(0)
                    elif (j in self._link_lists[i]) and (self._link_capa[i][j] > 0):
                        self._shr_dist[i].append(1)
                    else:
                        self._shr_dist[i].append(1e6) # inf
            for k in range(self._node_num):
                for i in range(self._node_num):
                    for j in range(self._node_num):
                        if(self._shr_dist[i][j] > self._shr_dist[i][k] + self._shr_dist[k][j]):
                            self._shr_dist[i][j] = self._shr_dist[i][k] + self._shr_dist[k][j] 
        elif msg == "demand_change":
            # from light load to mid load, may be for heavy load in the future
            self._request_times = [[30], [30], [30], [30]]
        else:
            raise NotImplementedError

    '''
    calculating the Widest Path from s to t
    '''
    def calcWP(self, s, t):
        fat = [-1] * self._node_num
        WP_dist = [- 1e6] * self._node_num
        flag = [False] * self._node_num
        WP_dist[t] = 1e6
        flag[t] = True
        cur_p = t
        while flag[s] == False:
            for i in self._link_lists[cur_p]:
                #bandwidth = max(self._link_capa[i][cur_p] - self._link_usage[i][cur_p], 0)
                bandwidth = self._link_capa[i][cur_p] - self._link_usage[i][cur_p] # for testing
                if min(bandwidth, WP_dist[cur_p]) > WP_dist[i]:
                    WP_dist[i] = min(bandwidth, WP_dist[cur_p])
                    fat[i] = cur_p
            cur_p = -1
            for i in range(self._node_num):
                if flag[i]:
                    continue
                if cur_p == -1 or WP_dist[i] > WP_dist[cur_p]:
                    cur_p = i
            flag[cur_p] = True
    
        path = [s]
        cur_p = 0
        while path[cur_p] != t:
            path.append(fat[path[cur_p]])
            cur_p += 1
        return path
    
    
    '''
    calculating the Shortest Path from s to t
    '''
    def calcSHR(self, s, t):
        path = [s]
        cur_p = 0
        while path[cur_p] != t:
            tmp_dist = 1e6
            next_hop = None
            for i in self._link_lists[path[cur_p]]:
                if self._shr_dist[i][t] < tmp_dist and self._link_capa[path[cur_p]][i] > 0:
                    next_hop = i
                    tmp_dist = self._shr_dist[i][t]
            path.append(next_hop)
            cur_p += 1
        return path
    
    '''
    calculating the Bandwidth-Constrained Shortest Path
    '''
    def calcBCSHR(self, s, t, demand):
        fat = [-1] * self._node_num
        SHR_dist = [1e6] * self._node_num
        flag = [False] * self._node_num
        SHR_dist[t] = 0
        flag[t] = True
        cur_p = t
        
        while flag[s] == False:
            for i in self._link_lists[cur_p]:
                #bandwidth = max(self._link_capa[i][cur_p] - self._link_usage[i][cur_p], 0)
                bandwidth = self._link_capa[i][cur_p] - self._link_usage[i][cur_p] 
                if bandwidth >= demand and SHR_dist[i] > SHR_dist[cur_p] + 1:
                    SHR_dist[i] = SHR_dist[cur_p] + 1
                    fat[i] = cur_p
            cur_p = -1
            for i in range(self._node_num):
                if flag[i]:
                    continue
                if cur_p == -1 or SHR_dist[i] < SHR_dist[cur_p]:
                    cur_p = i
            if SHR_dist[cur_p] < 1e6:
                flag[cur_p] = True
            else:
                break
    
        if not flag[s]:
            return None
        path = [s]
        cur_p = 0
        while path[cur_p] != t:
            path.append(fat[path[cur_p]])
            cur_p += 1
        return path

    '''
    method = "SHR"/"WP"/"DS"(diff-serv)
    '''
    def step_baseline(self, method):
        if method == "SHR":
            path = self.calcSHR(self._request.s, self._request.t)
        elif method == "WP":
            path = self.calcWP(self._request.s, self._request.t)
        elif method == "DS":
            if self._request.rtype == 0 or self._request.rtype == 3:
                path = self.calcSHR(self._request.s, self._request.t)
            elif self._request.rtype == 1:
                path = self.calcWP(self._request.s, self._request.t)
            else:
                path = self.calcBCSHR(self._request.s, self._request.t, self._request.demand)
                if path == None:
                    path = self.calcWP(self._request.s, self._request.t)
        elif method == 'QoS':
            path = self.calcBCSHR(self._request.s, self._request.t, self._request.demand)
            if path == None:
                path = self.calcWP(self._request.s, self._request.t)
        else:
            raise NotImplementedError
        self._request.path = copy.copy(path)

        # update link usage according to the selected path 
        capacity = 1e9
        for i in range(len(path) - 1):
            capacity = min(capacity, max(0, self._link_capa[path[i]][path[i + 1]] - self._link_usage[path[i]][path[i + 1]]))
            self._link_usage[path[i]][path[i + 1]] += self._request.demand
        count = len(path) - 1
        
        # install rules for path and generate service request in sim-env(ryu + mininet)
        ret_data = self.sim_interact(self._request, path)
        delay = ret_data['delay']
        throughput = ret_data['throughput']
        loss_rate = ret_data['loss']
        
        delta_dist = count - self._shr_dist[self._request.s][self._request.t]
        delta_demand = min(capacity, self._request.demand) / self._request.demand
        throughput_rate = throughput / self._request.demand
        rtype = self._request.rtype
        
        max_link_util = 0.
        for i in range(self._node_num):
            for j in range(self._node_num):
                if self._link_capa[i][j] > 0:
                    max_link_util = max(max_link_util, self._link_usage[i][j] / self._link_capa[i][j])
        print("max link utility:", max_link_util)

        # generate new state
        self._time_step += 1
        self._update_state()
        return rtype, delta_dist, delta_demand, delay, throughput_rate, loss_rate


if __name__ == "__main__":
    toponame = sys.argv[1]
    method = sys.argv[2] # baseline method can be SHR | WP | DS | QoS
    num_step = int(sys.argv[3])
    if toponame == "Abi":
        demand_matrix = "Abi_500.txt"
    else:
        demand_matrix = "GEA_500.txt"

    # setup env
    args = None 
    envs = NetEnv(args) 
    num_agent, num_node, observation_spaces, action_spaces, num_type = envs.setup(toponame, demand_matrix)
    envs.reset()
    
    # open log file
    log_dir = "../log/%s_%s_%d_simenv_heavyload_1-5loss/" % (toponame, method, num_step)
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.log'))
        for f in files:
            os.remove(f)
    log_dist_files = []
    log_demand_files = []
    log_delay_files = []
    log_throughput_files = []
    log_loss_files = []
    for i in range(num_type):
        log_dist_file = open("%s/dist_type%d.log" % (log_dir, i), "w", 1)
        log_dist_files.append(log_dist_file)
        log_demand_file = open("%s/demand_type%d.log" % (log_dir, i), "w", 1)
        log_demand_files.append(log_demand_file)
        log_delay_file = open("%s/delay_type%d.log" % (log_dir, i), "w", 1)
        log_delay_files.append(log_delay_file)
        log_throughput_file = open("%s/throughput_type%d.log" % (log_dir, i), "w", 1)
        log_throughput_files.append(log_throughput_file)
        log_loss_file = open("%s/loss_type%d.log" % (log_dir, i), "w", 1)
        log_loss_files.append(log_loss_file)

    for i in range(num_step):
        print("step:", i)
        rtype, delta_dist, delta_demand, delay, throughput_rate, loss_rate = envs.step_baseline(method)
        print(delta_dist, file=log_dist_files[rtype])
        print(delta_demand, file=log_demand_files[rtype])
        print(delay, file=log_delay_files[rtype])
        print(throughput_rate, file=log_throughput_files[rtype]) 
        print(loss_rate, file=log_loss_files[rtype])
