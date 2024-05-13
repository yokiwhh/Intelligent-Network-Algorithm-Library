from mininet.net import Mininet
from mininet.node import Controller, RemoteController, OVSSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.link import Link, Intf, TCLink
from mininet.topo import Topo
from mininet.util import custom, pmonitor
import logging
import os
from functools import partial
import socket
import json
import time
import heapq
import sys

class CustomTopo(Topo):
#自定义创建拓扑
    def __init__(self, nodeNum, linkSet, bandwidths, losses, **opts):
        Topo.__init__(self,**opts)
        self.__nodenum = nodeNum
        self.__linkset = linkSet
        self.__bandwidths = bandwidths
        self.__losses = losses

        self.__switches = []
        self.__hosts = []

        self.create_net()  #构建交换机及中间的link和相应参数
        self.add_hosts()   #构建host与对应交换机相连

    '''create the network topo'''
    def create_net(self):
        for i in range(self.__nodenum): #循环遍历每个节点，为其添加交换机
            self.__switches.append(self.addSwitch("s" + str(i + 1)))
            #addSwitch是Topo自带的函数，用于在拓扑中添加“s1、s2”等交换机，这里添加n（节点数）个交换机
        for i in range(len(self.__linkset)): #为每个交换机建立对应的边
            node1 = self.__linkset[i][0]
            node2 = self.__linkset[i][1]
            self.addLink(self.__switches[node1], self.__switches[node2], bw=self.__bandwidths[i], delay='5ms', loss=self.__losses[i], max_queue_size=1000) 
    
    '''add host for each switch(node)'''
    def add_hosts(self):
        if self.__nodenum >= 255:
            print("ERROR!!!")
            exit()
        for i in range(self.__nodenum):
            self.__hosts.append(self.addHost("h" + str(i + 1), mac=("00:00:00:00:00:%02x" % (i + 1)), ip = "10.0.0." + str(i + 1)))
            self.addLink(self.__switches[i], self.__hosts[i], bw=1000, delay='0ms') # bw here should be large enough
        

def generate_request(net, src, src_port, dst, dst_port, rtype, demand, rtime, time_step):
#针对从DRL客户端连接接受的消息生成请求
#rtime：回复时间
    TIME_OUT = 5
    src_host = net.hosts[src]
    dst_host = net.hosts[dst]

    popens = {}
    popens[dst_host] = dst_host.popen("python3 server.py %s %d %d %d %d" % (dst_host.IP(), dst_port, rtime, rtype, time_step))
    #就是生成相应str对应的1个进程，放到popens[dst_host]里面
    #向服务器传入：目的主机IP、目的端口...
    time.sleep(0.1)
    popens[src_host] = src_host.popen("python3 client.py %s %d %s %d %d %d %d" % (dst_host.IP(), dst_port, src_host.IP(), src_port, demand, rtime, time_step))
    #向客户端传入：目的主机IP、目的端口、源主机IP、源端口、需求...
    src_popen = popens[src_host]
    dst_popen = popens[dst_host]
    ind = 0
    time_stamp = time.time()  #time_stamp记录当前时间
    for host, line in pmonitor(popens):   #每次监控主机的popen中的一行
        if time.time() - time_stamp > TIME_OUT:  #这么长时间都没break，说明这个包丢了
            print("Request:", "src:", src, "dst:", dst, "rtype:", rtype, "demand:", demand)
            delay = TIME_OUT * 1000
            throughput = 0
            loss = 1.
            print("time out!")
            break
        if host:
            print("<%s>: %s" % (host.name, line))
            
            if host == dst_host:
                ret = line.split()
                delay = float(ret[1])
                throughput = float(ret[4])
                loss = float(ret[7])
                #flag = True
                if ind == 1: # avoid using the first data received from server
                    break
                else:
                    ind += 1
            
    return delay, throughput, loss, (src_popen, dst_popen)

def load_topoinfo(toponame):
#从Abi或者GEA中加载topo信息
    topo_file = open("./topo_info/%s.txt" % toponame, "r")
    content = topo_file.readlines()
    nodeNum, linkNum = map(int, content[0].split())
    linkSet = []
    bandwidths = []
    losses = []
    for i in range(linkNum):
        u, v, w, c, loss = map(int, content[i + 1].split())
        #u是出节点，v是入节点，c是带宽，loss是损失
        #1 4 1176 9920 0
        linkSet.append([u - 1, v - 1])
        bandwidths.append(float(c) / 1000)  # 链路的数据速率，Abi其中有1个瓶颈链路
        losses.append(loss)
    return nodeNum, linkSet, bandwidths, losses

if __name__ == '__main__':
    print "testbed initializing ..."
    toponame = sys.argv[1]
    #接受从run.sh中传入的参数Abi
    #sys.argv[0]=“testbed.py”
    if toponame == "test":
        nodeNum = 4
        linkSet = [[0, 1], [1, 2], [2, 3], [0, 3]]
        bandwidths = [1, 5, 5, 5]
        losses = [0, 0, 0, 0] # 0% must be int
    else:
        nodeNum, linkSet, bandwidths, losses = load_topoinfo(toponame)
    print "topoinfo loading finished."
    requests_pq = [] # put the popens of requests' server and client process
    #requests_pq存放发送请求的服务器或客户端进程
    
    topo = CustomTopo(nodeNum, linkSet, bandwidths, losses)  #根据info创建网络拓扑
    CONTROLLER_IP = "127.0.0.1" # Your ryu controller server IP
    CONTROLLER_PORT = 5001 
    OVSSwitch13 = partial(OVSSwitch, protocols='OpenFlow13') #返回OVSSwitch（OpenFlow13）
    net = Mininet(topo=topo, switch=OVSSwitch13, link=TCLink, controller=None)
    net.addController('controller', controller=RemoteController, ip=CONTROLLER_IP, port=CONTROLLER_PORT)
    net.start()  #相当于命令行中按照topo建立拓扑，并且根据ryu的openflow监听端口配置，连接至远端的ryu控制器；开启网络

    
    # build communication with DRL client
    print "waiting to simenv"
    # If using unique server for testbed, set TCP_IP to the server IP 
    TCP_IP = "127.0.0.1"
    TCP_PORT = 5000
    BUFFER_SIZE = 1024
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #s.settimeout(CHECK_TIMEOUT)
    #s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((TCP_IP, TCP_PORT))
    s.listen(1)    # s是监听的socket
    #1表示mininet的testbed只允许一个请求用户在队列里等待
    
    conn, addr = s.accept()   #从simenv接受的请求socket连接和连接地址
    # conn是已完成监听，用来传输数据的socket
    print('Connection address:', addr)
    time_step = 0
    # receive instruction from sim_env.py and generate request and send results
    #从simenv中接受指令信息，然后生成请求和发送结果
    while True:
        try:
            msg = conn.recv(BUFFER_SIZE)    #连接的消息实体，接受最大1024的内容
        except:
            sock.close()
            break
        #print("msg:", msg)

        #这一段暂时没看懂，队列里面有东西且第一个格小于time_step的时候，弹出堆顶，然后杀死两个进程？？？
        #堆顶是弹出最小值
        while len(requests_pq) > 0 and requests_pq[0][0] <= time_step:
            ind, popens = heapq.heappop(requests_pq)
            popens[0].kill()
            popens[1].kill()
        # 这里就是每次一个请求对产生client-
        

        data_js = json.loads(msg)   #msg是json格式的，这里将json格式数据转换为字典格式
        rtime = data_js['rtime']  #实际上没有将rtime作为参数传入生成请求函数
        delay, throughput, loss, popens = generate_request(net, data_js['src'], data_js['src_port'], data_js['dst'], data_js['dst_port'], data_js['rtype'], data_js['demand'], 1000000, time_step) # rtime is a deprecated para
        # rtype 应该是流的类型
        # demand 流的最大数据速率需求

        heapq.heappush(requests_pq, (rtime + time_step, popens))
        
        ret = {
                'delay': delay,
                'throughput': throughput,
                'loss': loss,
                }
        
        # For Abi link failure & demand change test
        # we let testbed send the failure information to simenv for simple implementation 
        if time_step == 10000:
            '''
            # link failue
            net.configLinkStatus('s1', 's5', 'down')
            ret['change'] = 'link_failure'
            '''
            # demand change
            #ret['change'] = "demand_change"
        
        
        msg = json.dumps(ret)
        conn.send(msg.encode())
        time_step += 1   # time_step处理的从simenv来的请求个数


    CLI(net)  # 显示网络链接情况
    

