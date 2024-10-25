# 1 .UR5_TCP_grip_for_onrobot

要求：

python3

执行文件：
TCPClient.py

使用python通过TCP使通讯指令onrobot和ros中间切换

由于onrobot夹爪没法直接与ros通讯，所以本脚本在URCAP上创建两个任务（关闭和开启），然后通过TCP向其发送执行指令:

    def tcp_rg2_close(self):
        message ="load close_rg2.urp\r\n"
        self.tcp_send(message)

    def tcp_rg2_open(self):
        message ="load open_rg2.urp\r\n"
        self.tcp_send(message)
        
完成控制后返回ros控制端：

    def tcp_ROS(self):
       message ="load external.urp\r\n"
       self.tcp_send(message)

       当然也可以用脚本直接控制夹爪和ur5，后续整理

# 2 .ur5moverealrobot

要求：

python3
pybullet
serial

执行文件：
ur5moverealrobot.py

代码有三个部分组成：
pybullet部分
ur5通讯控制部分
力控传感器通讯部分
完整的demo2涉及三个部分，其它的对应取用
