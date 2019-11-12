[//]: # (Image References)

[image1]: learning_curve.jpg "Learning Curve"
[image2]: learning_curve_3000episodes.jpg "Learning Curve 3000 episodes"


# Project 3: Collaboration and Competition

# Report

## The Learning Algorithm
I implemented the Multi-Agent Deep Deterministic Policy Gradient ([MA-DDPG](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf)) algorithm to train the agent. MADDPG is a multi-agent policy gradient algorithm where agents learn a centralized critic based on the observations and actions of all agents.

My implementation uses to following hyperparameter values:

* Replay Memory capacity: 1E6 experiences
* Batch size: 128 experiences
* gamma: 0.99
* tau: 0.05
* Learning Rate (actor): 0.0001
* Learning Rate (critic): 0.0005
* one learning step for every one experiences collected: per episode, each player generates one experience and executes one learning step

## The neural network models

Each player is a DDPG agent, which has an Actor and a Critic, and both the actor and critic have a "local" and a "target" neural network.


The actor networks are a typical neural network consisting each of three fully connected layers. They have 48 input nodes (two times the state size, 24, for each player) and two output nodes (the action size of a player). The two hidden layers have 256 and 128 nodes, respectively, and a ReLU activation function. The output layer has a tanh function so that the output range is [-1, +1].

The Critic networks are a classic neural network consisting each of three fully connected layers. The two hidden layers have a ReLU activation function. The first layer has 48 input nodes (two times the state size, 24, for each player) and 256 hidden nodes. The output of the this layer is then concatenated with the actions of each player. So the input to the second layers has 260 nodes: 256 from the first layer input and two each for the player actions. The second layers has 128 hidden nodes. The output layer has one output.

The weights of the target networks are updated using soft updated from the local networks.

## The Learning Curve

I ran several experiments with the models, to see the impact of different hyperparameter values. Most experiments showed little or no learning, and were aborted before the target score was reached.

Below is the learning curve for the hyperparameters shown above. The MADDPG algorithm ran for 3000 episodes, taking 10+ hours on my iMac (CPU only). The environment was solved in 843 episodes. The score kept improving and peaked at about 1850 episodes.

![Learning Curve][image1]

The curve below shows the scores and moving average for a much longer run (3000 steps). Unfortunately the way I calculate the score may not have been correct. However, the chart shows how the model performance goes up and down.

![Learning Curve][image2]


## Further Improvements

Algorithmic-wise, the [MA-DDPG](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf) authors believe the performance of MA-DDPG could be further improved by training with an ensemble of policies.

Coding-wise, it would be great to improve the generation of pseudo random numbers . Although I set the seed for the pseudo random generator (PRG) to a fixed value, somehow the computed scores change, and they shouldn't. I must not be doing it right. This makes it impossible to make reproducible experiments.