#coding=utf-8
import socket 
import sys
import time

BUFFER_SIZE = 128
TIME_OUT = 5

if __name__== '__main__': 
    addr = sys.argv[1]   # 目的主机地址
    port = int(sys.argv[2])  # 目的主机端口
    rtime = int(sys.argv[3])
    rtype = int(sys.argv[4])

    time_step = int(sys.argv[5]) # not used

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # 为目的主机建立socket
    sock.bind((addr, port))
    
    ind = 0
    delay = 0
    throughput = 0
    time_stamp = time.time()
    start_time = time.time()
    if rtype == 0:   # rtype是用来标注流的类型的,标记大小流的,CSTEP是应该是指不同大小的流处理的包间隔
        CSTEP = 5  # small flow only need delay, small CSTEP can speed up the experiment
        # 比如小流就5个包5个包处理就行
    elif rtype == 3:
        CSTEP = 50
    else:
        CSTEP = 30
    ind_stamp = 0  # 按照时间间隔处理流信息，上一次处理的包的序号戳
    while True:
        try:
            sock.settimeout(TIME_OUT)  # 套接字上TIME_OUT毫秒不活动后设置为超时
            data, addr = sock.recvfrom(BUFFER_SIZE)  # 接收套接字数据和地址  #每来一个包接受一次数据
        except socket.timeout:
            continue
        # 例如：小流。5个包进行一次处理。每个包到来时delay加上这个包的delay，throughput更新同理。
        # ind就是包的序号，ind % CSTEP == 0 时进入处理程序，打印这个流（5个包）时间内平均delay、throughput和丢包率，然后全部置零，等下一轮
        infos = str(data.decode()).split(';')[:-1]   # 接受的信息
        delay += int(time.time() * 1000) - int(infos[1])  # 当前时间-包的出发时间
        throughput += BUFFER_SIZE * 8
        if ind % CSTEP == 0 and ind != 0:
            # since no packet disorder in simulate environment
            # 因为在模拟环境中没有分组混乱
            # in fact we only need the first several records too much record(about 1000 records) will crash popen buffer and make the server killed
            # 事实上，我们只需要前几条记录,太多的记录(大约1000条记录)将崩溃popen缓冲区，使服务器死亡
            if ind / CSTEP <= 10:
                print("delay: %f ms throughput: %f Kbps loss_rate: %f" % (delay / CSTEP, throughput / 1e3 / (time.time() - time_stamp), (int(infos[0]) - ind_stamp - CSTEP) / (int(infos[0]) - ind_stamp)), flush=True)
                # throughput: 因为要算Kbps，所以throughput/1000/时间间隔
                # 将内容刷新到popen和pmonitor
            delay = 0
            throughput = 0
            time_stamp = time.time()  # 下一次事件的时间戳从上一个包处理完起算
            ind_stamp = int(infos[0])  # 这一轮处理到了第ind_stamp个包
        ind += 1  # 包+1
    sock.close()

