[//]: # (Image References)

[image1]: images/ContinuousControl.gif "Trained Agent"
<!-- [image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler" -->

# Continuous Control W/ Multi Agent DDQN

### Introduction

For this project, you will work with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Distributed Training

In this project there are two versions that can be used for solving:
- The first version contains a single agent.
- The second version contains 20 identical agents, each with its own copy of the environment.

Both copies can be downloaded in the getting started section.

The second version is useful for algorithms like [PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf), and [D4PG](https://openreview.net/pdf?id=SyZipzbCb) that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience.  

### Solving the Environment 

#### Option 1: Solve the First Version

The task is episodic, and in order to solve the environment,  your agent must get an average score of +30 over 100 consecutive episodes. An example of this running can be found in the notebook [here](done/Continuous_Control.ipynb) in the done folder (relevant checkpoints for this solve are there as well!)

#### Option 2: Solve the Second Version

The barrier for solving the second version of the environment is slightly different, to take into account the presence of many agents.  In particular, your agents must get an average score of +30 (over 100 consecutive episodes, and over all agents).  Specifically,
- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent.  This yields 20 (potentially different) scores.  We then take the average of these 20 scores. 
- This yields an **average score** for each episode (where the average is over all 20 agents).

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. 

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

    - **_Version 1: One (1) Agent_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

    - **_Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

### Instructions

Follow the instructions in `Continuous_Control.ipynb` to get started with training your own agent!

<!-- ### (Optional) Challenge: Crawler Environment

After you have successfully completed the project, you might like to solve the more difficult **Crawler** environment.

![Crawler][image2]

In this continuous control environment, the goal is to teach a creature with four legs to walk forward without falling.  

You can read more about this environment in the ML-Agents GitHub [here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#crawler).  To solve this harder task, you'll need to download a new Unity environment.  (**Note**: Udacity students should not submit a project with this new environment.)

You need only select the environment that matches your operating system:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Windows_x86_64.zip)

Then, place the file in the `p2_continuous-control/` folder in the DRLND GitHub repository, and unzip (or decompress) the file.  Next, open `Crawler.ipynb` and follow the instructions to learn how to use the Python API to control the agent.

(_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._) -->


## Training on AWS Deep Learning AMI (Ubuntu 16.04)

Note:
===

When using this environment it will cost money! Make sure to shut down and terminate the environment you create.

The expectation is to use the no vis environment for this training. Either the single agent or the 20 agent

You should also be able to run this network locally by executing the command

```bash
# passing params - used in completion version scores + 30 over 100 episodes
python ddqn.py --filename=app/Reacher_Linux_NoVis/Reacher.x86_64 --batch_size=1024 --update_every=10 --update_times=5 --lr_critic=0.0002 --lr_actor=0.0002 --weight_decay=0 --seed=2 --gamma=0.8
```

---

### Step 1: Choose AMI

Create a new instance in the EC2 panel and search for the AMI `ami-016ff5559334f8619` it can be found in region `us-east-1`

Select continue

##### OR

Follow the build instruction here: [Training on Amazon Web Service](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)

#### Step 2: Choose instance type

Select Family: `GPU Instances` Type: `p2.xlarge`

#### Step 3: Configure Instance

next

#### Step 4: Add Storage

next

#### Step 5: Add Tags

configure any tags you see fit

#### Step 6: Configure Security Group

allow ssh connections (default) - create new security group

#### Launch

create new keypair and download

when the instance is running connect to it using the pem file previously downloaded in `Step 6`

```bash
# change permissions on pem file
chmod 600 ~/.ssh/p2-xlarge-drl.pem

# connect
ssh -i ~/.ssh/p2-xlarge-drl.pem ubuntu@{hostname found in ec2 dashboard}
```

<!-- Start x server and use it
```bash
# Start the X Server, press Enter to come back to the command line
sudo /usr/bin/X :0 &

# Check if Xorg process is running
# You will have a list of processes running on the GPU, Xorg should be in the list.
nvidia-smi

# Make the ubuntu use X Server for display
export DISPLAY=:0
```

Ensure it is configured
```bash
# For more information on glxgears, see ftp://www.x.org/pub/X11R6.8.1/doc/glxgears.1.html.
glxgears
# If Xorg is configured correctly, you should see the following message

# Running synchronized to the vertical refresh.  The framerate should be
# approximately the same as the monitor refresh rate.
# 137296 frames in 5.0 seconds = 27459.053 FPS
# 141674 frames in 5.0 seconds = 28334.779 FPS
# 141490 frames in 5.0 seconds = 28297.875 FPS
``` -->

```bash
# clone environment from git
git clone https://github.com/kelstopper/drl_continuous_control.git && cd drl_continuous_control

# copy headless linux app YOU ONLY NEED 1 OF THE FOLLOWING
# WITH ONE AGENT
curl https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip > app/Reacher_Linux_NoVis.zip
#OR WITH 20 AGENTS
curl https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip > app/Reacher_Linux_NoVis.zip

cd app && unzip Reacher_Linux_NoVis.zip && cd ..

# use the pytorch env
source activate pytorch_p36
pip install unityagents

# if running for an extended time you may need to reconfigure sshd by adding the following to /etc/ssh/sshd_config
# ClientAliveInterval 300
# ClientAliveCountMax 2

# run the cnn example, verify that it is running on CUDA in the logs
## "Training on CUDA" <<< Should be present if "Training on CPU" is present you are training on cpu and it WILL take longer and cost more
python ddqn.py --filename=app/Reacher_Linux_NoVis/Reacher.x86_64 --batch_size=1024 --update_every=10 --update_times=5 --lr_critic=0.0002 --lr_actor=0.0002 --weight_decay=0 --seed=2 --gamma=0.8
```

Run the trained environment locally

```bash
# DOWNLOAD INSTRUCTIONS
download the saved png with the scores
download the saved network weights
```
