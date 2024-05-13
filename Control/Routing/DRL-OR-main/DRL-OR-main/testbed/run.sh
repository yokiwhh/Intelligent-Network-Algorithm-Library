sudo mn -c     '''在非mininet终端下，执行该命令，可以clear上一次退出topo所遗留的文件。'''
for((i=1;i<=100;i++)) do sudo lsof -i:5000 | awk '{print $2}' | awk 'NR==2{print}' | xargs kill -9; done
#“|”管道：把左侧命令的输出作为右侧命令的输入
#显示打开5000端口（mininet_host的监听端口）的所有进程
#一行一行的读取指定的文件，以空格作为分隔符，打印第二个字段（选取并打印第二列数据）
#awk 'NR==2{print}' 显示进程信息的第二行的内容
#xargs将接收到的第二行内容通过空格分割，作为kill -9（强制杀死进程）的参数（需要杀死进程的进程号）
#把之前的进程全都杀死
sudo python testbed.py Abi
#调用testbed文件并传入参数Abi
#sudo python testbed.py GEA
