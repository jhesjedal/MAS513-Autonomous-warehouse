#!/usr/bin/env python
import csv
import sys
import rospy
import moveit_commander
import geometry_msgs.msg
import os
import math
from tf.transformations import *


os.chdir('/home/bjj/catkin_ws/src/scripts/src/')

with open('Coord.csv','r') as cord:
    csv_reader = csv.reader(cord)
    for line in csv_reader:
        x1 = float(line[0])
        y1 = float(line[1])
        z1 = float(line[2])
        print(x1,y1,z1)

moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node('move_group_python_interface_tutorial', anonymous=True)
robot = moveit_commander.RobotCommander()

arm_group = moveit_commander.MoveGroupCommander("manipulator")
# Put the arm in the --- position
arm_group.set_named_target("home")
plan1 = arm_group.go()

pose_goal = geometry_msgs.msg.Pose()
pose_goal.position.x = x1
pose_goal.position.y = y1
pose_goal.position.z = z1
pose_goal.orientation.w = 1.0
pose_goal.orientation.x = 1.0
pose_goal.orientation.y = 1.0
pose_goal.orientation.z = 1.0

arm_group.set_pose_target(pose_goal)
plan1 = arm_group.go()

rospy.sleep(1)
rospy.signal_shutdown()

