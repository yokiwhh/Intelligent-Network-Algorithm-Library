#coding=utf-8
import socket
import sys
import time

BUFFER_SIZE = 128

def padding_bytes(x, target_len):
    clen = len(x)
    x += bytes(target_len - clen)
    return x

if __name__ == '__main__':
    server_addr = sys.argv[1]
    server_port = int(sys.argv[2])
    client_addr = sys.argv[3]
    client_port = int(sys.argv[4])
    demand = int(sys.argv[5])  # Kbps   # 最大数据速率需求
    rtime = int(sys.argv[6])  # seconds

    time_step = int(sys.argv[7])  # for testing
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((client_addr, client_port))    # 建立客户端的socket
    ind = 0
    start_time = time.time()
    time_stamp = int(time.time() * 1000)
    while True:  # 源主机不断向目的主机发送数据包
        temp_stamp = time.time()
        msg = "%d;%d;" % (ind, int(temp_stamp * 1000))  # 包序号，当前时间*1000（sever.py里面计算delay）
        msg = padding_bytes(msg.encode(), BUFFER_SIZE)
        sock.sendto(msg, (server_addr, server_port))
        
        ind += 1
        curr_bit = ind * BUFFER_SIZE * 8
        temp_stamp = time.time()
        if curr_bit > (temp_stamp - start_time) * demand * 1000:  # 超过最大需求速率就休息会
            time.sleep(BUFFER_SIZE / (demand * 125))
    sock.close()
