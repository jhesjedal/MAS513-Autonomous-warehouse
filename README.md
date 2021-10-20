# Computer vision based pick and place with maniputator


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
sudo apt-get update
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

### Launching Rviz Demo with the tm1100 config
	roslaunch tm1100_moveit_config demo.launch

### Launching Gazebo
	roslaunch tm1100_moveit_config gazebo.launch
However this leads to some errors from the moveit configuration files.

