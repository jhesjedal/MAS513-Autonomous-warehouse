# Computer vision based pick and place with maniputator

This repository is created for a project where a manipulator (Omron TM14/tm1100) does pick and place from computer vision data.

The repository is based on https://github.com/viirya/ros-driver-techman-robot which do not have a MoveIt configuration for this projects manipulator. A configuration for the tm1100 was made. For further use of already excisting content, see link above.

ROS version Melodic is implemented in this repository. If using a different ROS version, exchange "melodic" with the used ROS version.

# Installing_ROS
INSTALLING & GETTING STARTED WITH ROS | How to install ROS & How to setup Catkin Workspace on Ubuntu

The below commands are for installing ROS melodic on Ubuntu 18.04 </br>


### Setup sources.list
```
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
```

### Adding Key
```
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
```

### Update package list
```
sudo apt-get update && upgrade
```

### Installing ROS Melodic Full Desktop Version
```
sudo apt-get install ros-melodic-desktop-full
```

### Initialize Ros Dependencies
```
sudo rosdep init
```
```
rosdep update
```

### Setting up ROS Environment
```
printf "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
```
```
source ~/.bashrc
```

### Installing Python Packages for ROS
```
sudo apt-get install python-rosinstall
```
```
sudo apt install python-catkin-tools
```

### Other Important ROS Packages
```
sudo apt-get install ros-melodic-tf-*
```
```
sudo apt-get install ros-melodic-pcl-msgs ros-melodic-mav-msgs ros-melodic-mavros ros-melodic-octomap-* ros-melodic-geographic-msgs libgeographic-dev
```

### Creating Catkin Workspace
```
mkdir catkin_ws
```
```
cd catkin_ws
```
```
mkdir -p src
```
```
cd src
```
```
catkin_init_workspace
```
```
printf "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
```
```
cd ~/catkin_ws
```
```
catkin_make
```
```
source ~/catkin_ws/devel/setup.bash
```

### Checking ROS & Gazebo Versions
On Ubunutu 16.04, you should have **ROS Kinetic with Gazebo 7**, </br>
whereas on Ubuntu 18.04, you should have **ROS Melodic with Gazebo 9**
```
rosversion -d
```
```
gazebo -v
```

# Omron
For Omron repository and use of MoveIt see ROS-driver-techman-robot Repository:
https://github.com/viirya/ros-driver-techman-robot

### Installing nessesary controller for Rviz and gazebo simulation and dependencies.
	sudo  apt−get  install ros−melodic−joint−trajectory−controller
	sudo  apt−get  install ros−melodic−joint−trajectory−action

### Launching Rviz Demo with the tm1100 config
	roslaunch tm_gazebo tm1100_gazebo_moveit.launch

### Launching Gazebo
	roslaunch tm_gazebo tm1100.launch

### Executing script for motion
	Before executing th python scripts, it has to be executable. Making a python script executable is done with the following command:
	chmod +x importcsv.py
	
	Then the script can be executed:
	rosrun scripts importcsv.py
