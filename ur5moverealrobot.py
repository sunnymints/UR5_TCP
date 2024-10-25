import time
from math import *
from socket import *
import struct
import serial
import serial.tools.list_ports
from socket import *
import minimalmodbus as mm
import numpy as np
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation

'''
代码有三个部分组成：
pybullet部分
ur5通讯控制部分
力控传感器通讯部分
完整的demo2涉及三个部分，其它的对应取用
'''


class DebugAxes(object):
    """
    可视化某个局部坐标系, 红色x轴, 绿色y轴, 蓝色z轴
    """

    def __init__(self):
        self.uids = [-1, -1, -1]

    def update(self, pos, orn):
        """
        Arguments:
        - pos: len=3, position in world frame
        - orn: len=4, quaternion (x, y, z, w), world frame
        """
        pos = np.asarray(pos).reshape(3)

        rot3x3 = Rotation.from_quat(orn).as_matrix()
        axis_x, axis_y, axis_z = rot3x3.T
        self.uids[0] = p.addUserDebugLine(pos, pos + axis_x * 0.05, [1, 0, 0], replaceItemUniqueId=self.uids[0])
        self.uids[1] = p.addUserDebugLine(pos, pos + axis_y * 0.05, [0, 1, 0], replaceItemUniqueId=self.uids[1])
        self.uids[2] = p.addUserDebugLine(pos, pos + axis_z * 0.05, [0, 0, 1], replaceItemUniqueId=self.uids[2])


def relative_pos_and_ore_form_world(end_pos, end_orn, relative_offset, relative_euler):
    """
    目的：将该link下的相对移动和转动映射到绝对坐标下(当然，前两个值的父坐标系不是绝对坐标系，则以相关坐标系为准）
    Arguments:
    - end_pos: len=3, 该 link 的在世界坐标系的位置
    - end_orn: len=4, 该 link 的在世界坐标系的姿态 (x, y, z, w)
    - relative_offset 该 link下的相对移动 list of 3
    - relative_euler  该 link下的旋转  list of 3
    - Rotation.from_euler('XYZ', move_euler).as_matrix() 大写是内旋动轴旋转，小写相反

    注意：请注意该函数先旋转后移动，务必注意自己的变换要求！

    Returns:
    - wcT: shape=(4, 4), transform matrix, represents this link pose in world frame
    """

    end_orn = Rotation.from_quat(end_orn).as_matrix()
    wcT = np.eye(4)
    # wcT[:3, 3] = end_orn.dot(relative_offset) + end_pos #注意：务必注意自己的变换要求！
    fg = Rotation.from_euler('xyz', relative_euler).as_matrix()
    wcT[:3, :3] = np.matmul(end_orn[:3, :3], fg)
    wcT[:3, 3] = end_orn.dot(relative_offset) + end_pos

    return wcT


def fix_center_rotation(end_pos, end_orn, relative_offset, relative_euler, dy_M=0.09):
    """
       目的：固定点旋转
       Arguments:
       - end_pos: len=3, 该 link 的在世界坐标系的位置
       - end_orn: len=4, 该 link 的在世界坐标系的姿态 (x, y, z, w)
       - relative_offset 该 link下的相对移动 list of 3
       - relative_euler  该 link下的旋转  list of 3
       用法：eeink_link_next_Rotation_matrix = fix_center_rotation(self.current_pos, self.current_orie, [-0.05,0,0], [0, 0.3, 0])
       -[-0.05,0,0] eelink与固定轴端面的距离(X,Y,Z)
       - [0, 0.3, 0] 绕固定轴端面的旋转欧拉角(x,y,z)

       Returns:
       - eeink_link_next_Rotation_matrix: shape=(4, 4), transform matrix, represents this link next pose in world frame
       """
    # 前置变换：先平移eelink

    # peg_link_Rotation_matrix=relative_pos_and_ore_form_world(end_pos, end_orn, relative_offset, [0, 0, 0])#只平移不旋转
    # peg_link_Quaternion = Rotation.from_matrix(peg_link_Rotation_matrix[:3, :3]).as_quat()
    # peg_link_pos = peg_link_Rotation_matrix[:3, -1]

    # 1.先将eelink变换到peg末端坐标系,再平移相对位置
    peg_link_Rotation_matrix = relative_pos_and_ore_form_world(end_pos, end_orn,
                                                               [relative_offset[0] + dy_M, relative_offset[1] + 0,
                                                                relative_offset[2] + 0], [0, 0, 0])  #
    peg_link_Quaternion = Rotation.from_matrix(peg_link_Rotation_matrix[:3, :3]).as_quat()
    peg_link_pos = peg_link_Rotation_matrix[:3, -1]

    # 2.eelink在peg坐标系做完旋转
    eeink_link_next_Rotation_matrix = relative_pos_and_ore_form_world(peg_link_pos, peg_link_Quaternion, [0, 0, 0],
                                                                      relative_euler)
    # 3.通过相对坐标系平移变换，将peg坐标下的eelink坐标映射回真实eelink坐标
    eeink_link_next_Rotation_matrix = relative_pos_and_ore_form_world(eeink_link_next_Rotation_matrix[:3, -1],
                                                                      Rotation.from_matrix(
                                                                          eeink_link_next_Rotation_matrix[:3,
                                                                          :3]).as_quat(),
                                                                      [-dy_M, 0, 0], [0, 0, 0])

    return eeink_link_next_Rotation_matrix


class Model_robot():
    metadata = {'render.modes': ['human']}
    def __init__(self, render=True):
        super().__init__()

        self.Fusb = Forceusb()
        self.ft_compenstate= np.array([0,0,0,0,0,0])
        self.Visualize_rotation_center_UI = DebugAxes()  # 可视化旋转中心
        # 机械臂实际执行频率，ur5真实通讯频率是120hz
        self._timeStep = 120
        self.t = (1 / self._timeStep) * 2
        # --------------------------- 重置关节至初始状态--------------------------------
        self.init_joint_val = [0.29572491999307804, -1.3445488199311688, -2.1969757544996567,
                               -1.1709496137703805, 1.572643027403216, 0.29166692172730446]
        # 关节跳跃值
        self.joint_damping = [0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001]

        # 自带数据库地址
        self.urdf_root_path = pybullet_data.getDataPath()

        # 第一步，连接仿真环境
        self.is_render = render
        if self.is_render:
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)
        # 设定界面显示视角
        p.resetDebugVisualizerCamera(cameraDistance=1.5,
                                     cameraYaw=0,
                                     cameraPitch=-40,
                                     cameraTargetPosition=[0.55, -0.35, 0.2])

        # -----------------------------------------------------------------------添加模型-----------------------------------------------------------------------------------------------
        # 添加pybullet的额外数据地址，使程序可以直接调用到内部的一些模型
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # 添加桌子模型
        self.table_id = p.loadURDF("table/table.urdf", basePosition=[-1.01, 0, -0.315 - 0.02])

        self.ur5_id = p.loadURDF(
            "/home/wgh/Downloads/qq-files/729566915/file_recv/real_example20240518/search_phase/F2tiaoshi_last/F2tiaoshi_last/ur_description/urdf/ur5_robot_sensor_pos2_stand_eelink.urdf",
            basePosition=[0, 0, 0.1], flags=9)
        for j in range(p.getNumJoints(self.ur5_id)):
            # print(j,p.getJointState(self.ur5_id,j))
            print(j, p.getJointInfo(self.ur5_id, j))



    # 获取机器人受力
    def getForceTorque(self):
        """
        获取机器人当前关节力/力矩
        """

        data_sub = 3
        FT_o = np.zeros([data_sub, 6])

        for i in range(data_sub):
            FT_o[i, :] = np.array(self.Fusb.get_current_ft(), dtype=float).reshape(1, 6)

        f_T = np.mean(FT_o[1:], axis=0)
        force = f_T[0:3]
        torque = f_T[3:]
        d = 0.042 - 0.0034
        FT = [-int(force[2] * 10) / 10, -int(force[0] * 10) / 10, -int(force[1] * 10) / 10,
              int(torque[2] * 100) / 100, -int((torque[0] - force[1] * d) * 100) / 100,
              -int((torque[1] + d * force[0]) * 100) / 100]
        print(FT)
        return FT

    #
    def get_compensate_ft(self):
        """
        力传感器连续零漂处理（导纳控制中使用）
        """
        FT = self.getForceTorque()
        print('get_compensate_ft中的FT：',FT)

        FT = [FT[0] - self.ft_compenstate[0], FT[1] - self.ft_compenstate[1], FT[2] - self.ft_compenstate[2],
              FT[3] - self.ft_compenstate[3], FT[4] - self.ft_compenstate[4], FT[5] - self.ft_compenstate[5]]
        if FT[0] < 0:
            FT[0] = 0.0111
        return FT

    def reset_Force_sensor(self):
        '''传感器重新连接，自动置零'''
        A = True
        time.sleep(2)
        while A == True:
            try:
                self.Fusb.close()
                self.Fusb = Forceusb()
                A = False
            except:
                time.sleep(0.5)
                A = True

    # 位置导纳函数，使用movej
    def admittance_position(self, M_position, B_position, K_position):
        '''位置导纳'''
        # posture_new = p.getLinkState(self.ur5_id, 7)[5]
        m_position = []
        fext = self.getForceTorque()[0:3]  # 传感器端
        # fext = self.Dynamic_rotation_center()[0:3]  # 动态力矩中心
        if np.abs(np.array(fext)).max() < 1:
            m_position = [0, 0, 0]
            # self.ft_compenstate = self.getForceTorque()
        else:
            # print("fext", fext)
            for i in range(3):
                xd = self.solution(-fext[i], M_position, B_position[i], K_position)
                m_position.append(xd)

            m_position = [m_position[0], m_position[1], m_position[2]]
        return m_position

    #姿态导纳
    def admittance_posture(self, M_posture, B_posture, K_posture):
        posture = []
        # fext = self.get_compensate_ft()[3:6]  # 传感器端面
        f_t = self.getForceTorque()  # 传感器端
        fext = f_t[:3]
        torue = f_t[3:]
        # print('"var forceture:',fext )
        if np.abs(np.array(fext)).max() < 1:
            posture=[0,0,0]
        else:
            for i in range(3):
                xd = self.solution(-torue[i]*10, M_posture, B_posture, K_posture)
                posture.append(xd)

        posture_x = 0  # posture[0]
        posture_y = posture[1]
        posture_z = posture[2]
        m_posture = [posture_x, posture_y, posture_z]
        return m_posture

    # 导纳求解
    def solution(self, fext, M, B, K):

        xd = (B * fext / (2 * K * sqrt(B ** 2 - 4 * K * M)) - fext / (2 * K)) * exp(
            self.t * (-B - sqrt(B ** 2 - 4 * K * M)) / (2 * M)) + (
                     -B * fext / (2 * K * sqrt(B ** 2 - 4 * K * M)) - fext / (2 * K)) * exp(
            self.t * (-B + sqrt(B ** 2 - 4 * K * M)) / (2 * M)) + fext / K
        return complex(xd).real
    def go_speed(self, target_pos, target_orie):
        # ------------------------------------------求解器-------------------------------------------------------
        self.target_robot_joint_angles = p.calculateInverseKinematics(
            bodyUniqueId=self.ur5_id,
            endEffectorLinkIndex=7,
            targetPosition=target_pos,
            targetOrientation=target_orie,
            jointDamping=self.joint_damping, )

        for i in range(6):
            p.resetJointState(bodyUniqueId=self.ur5_id, jointIndex=i + 1, targetValue=self.target_robot_joint_angles[i])
        p.stepSimulation()

        p.setJointMotorControlArray(self.ur5_id, [1, 2, 3, 4, 5, 6],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=list(self.target_robot_joint_angles),
                                    forces=np.array([87.0, 87.0, 87.0, 87.0, 60, 60]))

        # ----------------------将更新频率设置为真实频率---------------------
        p.setTimeStep(1.0 / self._timeStep)
        for _ in range(25):
            p.stepSimulation()

        # robot_jiont_states = np.zeros([6])
        # for j in [1, 2, 3, 4, 5, 6]:
        #     robot_jiont_states[j - 1] = p.getJointState(self.ur5_id, j)[0]
        '''连续获取真实机器人关节值有误，谨慎使用'''
        robot_jiont_states = np.array(real_robot_connect.UR_30003rt('q target')[1])
        speed_jiont = self.target_robot_joint_angles-robot_jiont_states
        real_robot_connect.speedj(speed_jiont)
        time.sleep(2.01)#等待时间略大于执行时间0.01s,如果想做到顺滑，需要轨迹插值，且等待时间小于执行时间，或者使用速度控制器（内部自己做了插值）

    def go(self, target_pos, target_orie):
        # ------------------------------------------求解器-------------------------------------------------------
        self.target_robot_joint_angles = p.calculateInverseKinematics(
            bodyUniqueId=self.ur5_id,
            endEffectorLinkIndex=7,
            targetPosition=target_pos,
            targetOrientation=target_orie,
            jointDamping=self.joint_damping, )

        for i in range(6):
            p.resetJointState(bodyUniqueId=self.ur5_id, jointIndex=i + 1, targetValue=self.target_robot_joint_angles[i])
        p.stepSimulation()

        p.setJointMotorControlArray(self.ur5_id, [1, 2, 3, 4, 5, 6],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=list(self.target_robot_joint_angles),
                                    forces=np.array([87.0, 87.0, 87.0, 87.0, 60, 60]))

        # ----------------------将更新频率设置为真实频率---------------------
        p.setTimeStep(1.0 / self._timeStep)
        for _ in range(25):
            p.stepSimulation()
        real_robot_connect.movej(self.target_robot_joint_angles)

        time.sleep(2.01)#等待时间略大于执行时间0.01s,如果想做到顺滑，需要轨迹插值，且等待时间小于执行时间，或者使用速度控制器（内部自己做了插值）


    def reset(self):

        # 安全设置
        safe_pose= [0.29480578842411714, -1.2777899899535896, -2.002758482598918, -1.4319075968715884, 1.5708793527071245, 0.2942708693688756]
        for i in range(6):
            p.resetJointState(bodyUniqueId=self.ur5_id, jointIndex=i + 1, targetValue=safe_pose[i])

        p.setJointMotorControlArray(self.ur5_id, [1, 2, 3, 4, 5, 6],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=list(safe_pose),
                                    forces=np.array([87.0, 87.0, 87.0, 87.0, 60, 60]))

        # ----------------------将更新频率设置为真实频率---------------------
        p.setTimeStep(1.0 / self._timeStep)
        for _ in range(25):
            p.stepSimulation()
        real_robot_connect.movej(safe_pose)
        time.sleep(2.1)

        self.reset_Force_sensor()





class Forceusb():
    '''
    modbus通讯，注册并获取力控数值
    有做零位减法，每次初始化就是置零的过程
    '''
    def __init__(self):
        self.BAUDRATE=19200
        self.BYTESIZE=8
        self.PARITY="N"
        self.STOPBITS=1
        self.TIMEOUT=0.2
        self.PORTNAME=self.serial_ports()
        self.SLAVEADDRESS=9
        self.ser=serial.Serial(port=self.PORTNAME, baudrate=self.BAUDRATE, bytesize=self.BYTESIZE, parity=self.PARITY, stopbits=self.STOPBITS, timeout=self.TIMEOUT)
        self.packet = bytearray()
        self.sendCount=0
        while self.sendCount<50:
            self.packet.append(0xff)
            self.sendCount=self.sendCount+1
        self.ser.write(self.packet)
        self.ser.close()
        #Communication setup
        mm.BAUDRATE=self.BAUDRATE
        mm.BYTESIZE=self.BYTESIZE
        mm.PARITY=self.PARITY
        mm.STOPBITS=self.STOPBITS
        mm.TIMEOUT=self.TIMEOUT
        self.ft300=mm.Instrument(self.PORTNAME, slaveaddress=self.SLAVEADDRESS)
        self.registers=self.ft300.read_registers(180,6)
        # Save measured values at rest. Those values are use to make the zero of the sensor.
        self.fxZero=self.forceConverter(self.registers[0])
        self.fyZero=self.forceConverter(self.registers[1])
        self.fzZero=self.forceConverter(self.registers[2])
        self.txZero=self.torqueConverter(self.registers[3])
        self.tyZero=self.torqueConverter(self.registers[4])
        self.tzZero=self.torqueConverter(self.registers[5])

        self.ft_data = []
        self.num = 0


    def close(self):
        self.ft300.serial.close()
    def serial_ports(self):#自动寻找端口
        ports = list(serial.tools.list_ports.comports())
        for port_no, description, address in ports:
            if 'USB' in description:
                return port_no


    def forceConverter(self,forceRegisterValue):
        """Return the force corresponding to force register value.

        input:
            forceRegisterValue: Value of the force register

        output:
            force: force corresponding to force register value in N
        """
        force=0
        forceRegisterBin=bin(forceRegisterValue)[2:]
        forceRegisterBin="0"*(16-len(forceRegisterBin))+forceRegisterBin
        if forceRegisterBin[0]=="1":
            #negative force
            force=-1*(int("1111111111111111",2)-int(forceRegisterBin,2)+1)/100
        else:
            #positive force
            force=int(forceRegisterBin,2)/100
        return force


    def torqueConverter(self,torqueRegisterValue):
        """Return the torque corresponding to torque register value.

        input:
            torqueRegisterValue: Value of the torque register

        output:
            torque: torque corresponding to force register value in N.m
        """
        torque=0

        torqueRegisterBin=bin(torqueRegisterValue)[2:]
        torqueRegisterBin="0"*(16-len(torqueRegisterBin))+torqueRegisterBin
        if torqueRegisterBin[0]=="1":
            #negative force
            # torque=-1*(int("1111111111111111",2)-int(torqueRegisterBin[1:],2)+1)/1000
            torque=-1*(int("1111111111111111",2)-int(torqueRegisterBin,2)+1)/1000
        else:
            #positive force
            torque=int(torqueRegisterBin,2)/1000
        return torque

    def SN(self,snValue):
        pass


    def get_current_ft(self):
        """
        获得力传感器数值
        """
        registers=self.ft300.read_registers(180,6)
        fx=round(self.forceConverter(registers[0])-self.fxZero,2)
        fy=round(self.forceConverter(registers[1])-self.fyZero,2)
        fz=round(self.forceConverter(registers[2])-self.fzZero,2)
        tx=round(self.torqueConverter(registers[3])-self.txZero,2)
        ty=round(self.torqueConverter(registers[4])-self.tyZero,2)
        tz=round(self.torqueConverter(registers[5])-self.tzZero,2)
        ft = [fx, fy, fz, tx, ty, tz]
        self.ft_data.append(ft)
        # print("ft", ft[3:6])
        return ft

# Fusb=Forceusb()
# force=Fusb.get_current_ft()
# print(force)

class socket_TCp_UR30003():
    def __init__(self):
        self.host_name = "192.168.1.102"
        self.port_num = 30003
        self.ClientSocket = socket(AF_INET,SOCK_STREAM)
        self.ClientSocket.connect((self.host_name,self.port_num))


    def UR_30003Script(self, send_data):
        # print(send_data)
        self.ClientSocket.send(send_data.encode('utf8'))


    def UR_30003rt(self,Meaning):

        dic= {'MessageSize': 'i', 'Time': 'd', 'q target': '6d', 'qd target': '6d', 'qdd target': '6d','I target': '6d',
            'M target': '6d', 'q actual': '6d', 'qd actual': '6d', 'I actual': '6d', 'I control': '6d',
            'Tool vector actual': '6d', 'TCP speed actual': '6d', 'TCP force': '6d', 'Tool vector target': '6d',
            'TCP speed target': '6d', 'Digital input bits': 'd', 'Motor temperatures': '6d', 'Controller Timer': 'd',
            'Test value': 'd', 'Robot Mode': 'd', 'Joint Modes': '6d', 'Safety Mode': 'd', 'empty1': '6d', 'Tool Accelerometer values': '3d',
            'empty2': '6d', 'Speed scaling': 'd', 'Linear momentum norm': 'd', 'SoftwareOnly': 'd', 'softwareOnly2': 'd', 'V main': 'd',
            'V robot': 'd', 'I robot': 'd', 'V actual': '6d', 'Digital outputs': 'd', 'Program state': 'd', 'Elbow position': '3d', 'Elbow velocity': '3d'}
        data=self.ClientSocket.recv(1220)
        ii=range(len(dic))
        for key,i in zip(dic,ii):
            fmtsize=struct.calcsize(dic[key])
            info,data=data[0:fmtsize],data[fmtsize:]
            fmt="!"+dic[key]
            dic[key]=dic[key],struct.unpack(fmt, info)
        f=1

        return dic[Meaning]



    def movej_offset(self,offset):
        '''TCP_pos:是当前tool的
        '''
        send_data = f'''
    def whf():
        set_tcp(p[0,0,0,0,0,0])
        global pose=get_actual_tcp_pose()
        global P= pose_trans(pose,p[{offset[0]},{offset[1]},{offset[2]},{offset[3]},{offset[4]},{offset[5]}])
        # global P2=get_inverse_kin(P)
        movej(P, a=0.05, v=0.25, t=0, r=0)
        

    end
        '''
        # print(send_data)
        self.UR_30003Script(send_data)#30003发送


    def movej(self,offset):
        '''TCP_pos:是当前tool的
        '''
        send_data = f'''
    def whf():
        set_tcp(p[0,0,0,0,0,0])
        # global pose=[{offset[0]},{offset[1]},{offset[2]},{offset[3]},{offset[4]},{offset[5]}]
        # popup(pose)
        movej([{offset[0]},{offset[1]},{offset[2]},{offset[3]},{offset[4]},{offset[5]}], a=0.05, v=0.25, t=1.8, r=0)
    

    end
            '''
        self.UR_30003Script(send_data)#30003发送 t=2.1
       
       
 #直接控制六个轴的速度
    def speedj(self,offset):
        send_data = f'''
    def whf():
        set_tcp(p[0,0,0,0,0,0])
        speedj([{offset[0]},{offset[1]},{offset[2]},{offset[3]},{offset[4]},{offset[5]}], 0.2,1)
    end
            '''
        self.UR_30003Script(send_data)#30003发送
        
#控制末端速度类似speedl，但是是欧拉角下  
    def speedj_offset(self,offset):

        send_data = f'''
    def whf():
        set_tcp(p[0,0,0,0,0,0])
        global pose=get_actual_tcp_pose()
        global P= pose_trans(pose,p[{offset[0]},{offset[1]},{offset[2]},{offset[3]},{offset[4]},{offset[5]}])
        global P2=get_inverse_kin(pose)
        global P3=get_inverse_kin(P)
        global P4=[P3[0]-P2[0],P3[1]-P2[1],P3[2]-P2[2],P3[3]-P2[3],P3[4]-P2
        [4],P3[5]-P2[5]]
        speedj(P4, 0.2,0.5)
    end
            '''
    # print(send_data)
        self.UR_30003Script(send_data)#30003发送

        # [-0.08972922650022963, -1.6219628747300388, -2.0232094758251753, -1.0464530101386091, 1.5717372302313297,
        #  -0.08981790491708695]




# R=socket_TCp_UR30003()
# # jionts=R.UR_30003rt('q target')
# # print(jionts)
# # import numpy as np
# # import math
# # np.multiply(np.array(jionts[1]),180.0000000/math.pi)
# R.movej_offset([-0.01,0,-0,-0,0,0])#写在类中global pose=get_actual_tcp_pose()不执行，所以与offest相关的函数必须写在类外，原因不清楚
# # R.movej([-0.08972922650022963, -1.6219628747300388, -2.0232094758251753, -1.0464530101386091, 1.5717372302313297,-0.08981790491708695])
#
# #test speedj
# # R.speedj_offset([0.0, 0.03,0,0.002,0,0])
# # R.speedj([0.0, 0.0,0,0.00,0,0.0])
#
# print('dd')

if __name__ == '__main__':


    '''demo1: force sensor usage for pybullet getForce fuction
        原始传感器信号与真实传感器坐标一致    
    '''
    # Fusb = Forceusb()
    # Fusb.close()
    # time.sleep(0.5)
    # Fusb = Forceusb()
    # # #
    # while True:
    #     F_T= np.array(Fusb.get_current_ft()[0:6], dtype=float).reshape(-1, 1)
    #     force = F_T[0:3]
    #     torque = F_T[3:6]
    #     time.sleep(0.05)
    #     print(F_T.reshape(-1))
    '''demo1': force sensor usage for pybullet getForce fuction
        将传感器信号进行一定处理和变换，与pybullet模型坐标一致
    '''
    # Fusb = Forceusb()
    # data_sub = 5
    # FT_o = np.zeros([data_sub, 6])
    # while True:
    #     for i in range(data_sub):
    #         FT_o[i, :] = np.array(Fusb.get_current_ft(), dtype=float).reshape(1, 6)
    #     f_T = np.mean(FT_o[1:], axis=0)
    #     force = f_T[0:3]
    #     torque = f_T[3:]
    #     d = 0.042 - 0.0034
    #     FT = [-int(force[2] * 10) / 10, -int(force[0] * 10) / 10, -int(force[1] * 10) / 10,
    #           int(torque[2] * 100) / 100, int((torque[0] - force[1] * d) * 100) / 100,
    #           int((torque[1] + d * force[0]) * 100) / 100]
    #     print(FT)
    #     time.sleep(0.5)
    '''demo2: force sensor usage for pybullet getForce fuction
        导纳控制
        K越小越敏感，B越小越敏感，M越小越敏感
    '''
    # real_robot_connect = socket_TCp_UR30003() #机械臂建立链接
    #
    # Robot = Model_robot()
    # Robot.reset()
    #
    # M_position=0.05
    # B_position = np.zeros(3)
    # B_position[0] =2
    # B_position[1] =2
    # B_position[2] =2
    # K_position = 5
    #
    # M_posture = 0.008
    # B_posture = 1
    # K_posture = 5
    #
    # Robot.ft_compenstate=Robot.getForceTorque()
    #
    # while True:
    #     # 获取当前的关节状态
    #     Robot.current_pos = p.getLinkState(Robot.ur5_id, 7)[4]
    #     Robot.current_orie = p.getLinkState(Robot.ur5_id, 7)[5]
    #     Robot.Visualize_rotation_center_UI.update(Robot.current_pos,Robot.current_orie)
    #     # m_position = Robot.admittance_position(M_position, B_position, K_position)
    #
    #     # eeink_link_next_Rotation_matrix = fix_center_rotation(Robot.current_pos, Robot.current_orie,
    #     #                                                       [m_position[0], m_position[1], m_position[2]],
    #     #                                                       [0,0,0],
    #     #                                                       dy_M=0.092)
    #
    #     m_posture = Robot.admittance_posture(M_posture, B_posture, K_posture)
    #     eeink_link_next_Rotation_matrix = fix_center_rotation(Robot.current_pos, Robot.current_orie,
    #                                                           [0,0,0],
    #                                                           [m_posture[0], m_posture[1], m_posture[2]],
    #                                                           dy_M=0.092)
    #
    #
    #
    #     # eeink_link_next_Rotation_matrix = fix_center_rotation(Robot.current_pos, Robot.current_orie,
    #     #                                                       [m_position[0], m_position[1], m_position[2]],
    #     #                                                       [m_posture[0], m_posture[1], m_posture[2]],
    #     #                                                       dy_M=0.092)
    #
    #
    #     end_Quaternion = Rotation.from_matrix(eeink_link_next_Rotation_matrix[:3, :3]).as_quat()
    #     end_pos1 = eeink_link_next_Rotation_matrix[:3, -1]
    #     # Robot.go(end_pos1,end_Quaternion)#仅仅实现位置控制，不是特别丝滑，需要进行插值或者用速度控制
    # '''连续获取真实机器人关节值有误，谨慎使用'''
    #     # Robot.go_speed(end_pos1,end_Quaternion)#速度控制（没有插值的速度控制也挺丝滑），但是TCP高频返回的关节值不能保证正确，这让人很疑惑（后续可以使用末端指令）








    '''demo3: robot state  pycharm 调试会出错'''
    R = socket_TCp_UR30003() #机械臂建立链接

    TOOL6D=R.UR_30003rt('Tool vector actual')  #取出反馈值Tool的位姿(笛卡尔6D位姿)
    print('Tool vector actual',TOOL6D)
    robot_jiont_states = R.UR_30003rt('q target')#取出关节状态
    print('robot_jiont_states', np.array(robot_jiont_states[1]))
    print('robot_jiont_states', np.array(robot_jiont_states[1])*180/np.pi)

    '''demo4: robot movej control '''
    # R = socket_TCp_UR30003() #机械臂建立链接
    # R.movej([0.294092870401073, -1.33090764529892, -2.171405356667104, -1.210179931805639, 1.572316109553702, 0.29055288564098003])
    # #























    '''robot jiont states rad to  *180/np.pi'''
    # print(np.array([0.29470252959723864, -1.3434761825351185, -2.1951895338764267, -1.1738231388222164, 1.5726177712823106, 0.2906822850770858])*180/np.pi)

    '''实验记录：pybullet姿态下对中的时的关节角度：'''
    #正好与方块表面贴合 (0.29423205642086486, -1.3442299295978728, -2.1957716121972233, -1.1724944331805487, 1.5726282071398505, 0.29019361321914783)
    #从方块表面保持安全距离 (0.294092870401073, -1.33090764529892, -2.171405356667104, -1.210179931805639, 1.572316109553702, 0.29055288564098003)



    # print('demo test')

("h2:0.37607478039689995, -7.807431919287485e-05, 0.2155089370929073, -2.2142314357490247, 2.2199948309453146, 0.02103291152049035"
 "h0；0.37608244026100573, -0.023916695305652443, 0.2664123943283833, -2.2143369560204276, 2.220005323432923, 0.021106454548129868"
)
'''0.2664123943283833-0.2155089370929073+0.0415=0.09240345723547602'''