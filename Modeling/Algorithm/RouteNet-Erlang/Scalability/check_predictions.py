import sys
import numpy as np
import networkx as nx
from glob import iglob
import re

sys.path.insert(1, "./code")
from read_dataset import input_fn, network_to_hypergraph
from model import GNN_Model
import configparser
import tensorflow as tf
from datanetAPI import DatanetAPI
import os
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
# set a grey background (use sns.set_theme() if seaborn version 0.11.0 or above)
sns.set(style="darkgrid")

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def transformation(x, y):
    traffic_mean = 666.4519976306121
    traffic_std = 418.79412331425846
    packets_mean = 0.660199595571597
    packets_std = 0.4204438794894145
    bandwidth_mean = 21166.35
    bandwidth_std = 24631.01
    scale_mean = 10.5
    scale_std = 5.77

    x["traffic"] = (x["traffic"] - traffic_mean) / traffic_std

    x["packets"] = (x["packets"] - packets_mean) / packets_std

    x["capacity"] = (x["capacity"] - bandwidth_mean) / bandwidth_std

    x["scale"] = (x["scale"] - scale_mean) / scale_std

    return x, y


params = configparser.ConfigParser()
params._interpolation = configparser.ExtendedInterpolation()
params.read('config.ini')

model = GNN_Model(params)

best = None
best_mre = float('inf')
for f in os.listdir('./ckpt_dir'):
    if os.path.isfile(os.path.join('./ckpt_dir', f)):
        reg = re.findall("\d+\.\d+", f)
        if len(reg) > 0:
            mre = float(reg[0])
            if mre <= best_mre:
                best = f.replace('.index', '')
                if '.data' in best:
                    idx = best.rfind('.')
                    best = best[:idx]
                best_mre = mre

print("BEST CHECKOINT FOUND: {}".format(best))
model.load_weights('./ckpt_dir/{}'.format(best))

directories = [d for d in iglob(params['DIRECTORIES']['test'] + '/*')]
# First, sort by scenario and second, by topology size

directories.sort(key=lambda f: (os.path.dirname(f), int(os.path.basename(f))))

path_MAPE = {}

num_samples = 0
for d in directories:
    ds_test = input_fn(d, min_scale=10, max_scale=11, shuffle=False)
    ds_test = ds_test.map(lambda x, y: transformation(x, y))
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    print("PREDICTING...")
    print("Directory: {}".format(d))
    scenario = int(str(os.path.basename(d)))
    pred = model.predict(ds_test)
    pred = np.squeeze(pred)

    print("COLLECTING...")
    tool = DatanetAPI(d, shuffle=False)
    it = iter(tool)
    index = 0
    for sample in it:
        num_samples += 1
        print(num_samples)

        G_copy = sample.get_topology_object().copy()
        T = sample.get_traffic_matrix()
        R = sample.get_routing_matrix()
        D = sample.get_performance_matrix()
        P = sample.get_port_stats()
        HG = network_to_hypergraph(network_graph=G_copy,
                                   routing_matrix=R,
                                   traffic_matrix=T,
                                   performance_matrix=D,
                                   port_stats=P,
                                   min_scale=10,
                                   max_scale=11)
        link_nodes = [n for n in HG.nodes if n.startswith('l_')]
        path_nodes = [n for n in HG.nodes if n.startswith('p_')]
        pred_delay = [{"pred_delay": occ} for occ in pred[index:index + len(path_nodes)]]
        delay_dict = dict(zip(path_nodes, pred_delay))
        nx.set_node_attributes(HG, delay_dict)

        for path in path_nodes:
            #print(HG.nodes[path]['pred_delay'])
            #print(HG.nodes[path]['delay'])
            HG.nodes[path]['MRE'] = ((HG.nodes[path]['pred_delay'] - HG.nodes[path]['delay']) / HG.nodes[path][
                'delay'])*100
            #print("MAPE: {:.2f} %".format(HG.nodes[path]['MRE']))
        index += len(path_nodes)

        MRE = []
        ground_truth = []
        gae = []
        for p in path_nodes:
            MRE.append(np.mean(np.abs(HG.nodes[p]['MRE'])))
            gae.append(HG.nodes[p]['MRE'])
            ground_truth.append(0)

        '''
        for p in path_nodes:
            neighbour_links = [n_l for n_l in HG[p]]
            l_mre = []
            for n_l in neighbour_links:
                l_mre.append(HG.nodes[n_l]['MRE'])
            MRE.append(np.mean(np.abs(l_mre)))
        '''

        print("MAPE: {} %".format(MRE))

    kwargs = {'cumulative': True}
    #sns.distplot(MRE, hist_kws=kwargs, kde_kws=kwargs,color = "r")
    #sns.distplot(ground_truth, hist_kws=kwargs, kde_kws=kwargs, color = "b")
    '''
    sns.kdeplot(MRE, cumulative=True, label='Distribution 1',color = "r")
    sns.kdeplot(gae, cumulative=True, label='Distribution 2',color = "b")
    plt.xlabel('Relative Error [(y − yˆ)/y × 100]')
    plt.ylabel('CDF')
    plt.legend(['RouteNet', 'Masknet'])
    plt.savefig('./delay.png', format='png')
    #sns.kdeplot(ground_truth, cumulative=True, label='Distribution 3', color="g")
    plt.show()
    '''

    sns.kdeplot(MRE, cumulative=True, label='Distribution 1', color='r', shade=True, lw=2)
    sns.kdeplot(gae, cumulative=True, label='Distribution 2', color='b', shade=True, lw=2)
    # 如果需要，可以添加ground_truth的数据
    # sns.kdeplot(ground_truth, cumulative=True, label='Distribution 3', color='g', shade=True)

    # 添加标题和轴标签
    #plt.title('CDF of Relative Error')
    plt.xlabel(r'$(y - \hat{y})/y \times 100$')
    plt.ylabel('CDF')

    # 添加图例
    plt.legend(['RouteNet-Erlang', 'Masknet'])

    # 保存图表
    plt.savefig('./gaeL.pdf', format='pdf')

    # 显示图表
    plt.show()


    '''
        if scenario not in path_MAPE:
            path_MAPE[scenario] = {}
        if len(G_copy) not in path_MAPE[scenario]:
            path_MAPE[scenario][len(G_copy)] = []
        path_MAPE[scenario][len(G_copy)].append(MRE)
    '''
