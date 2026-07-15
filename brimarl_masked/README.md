# Biscola RL
Since this code is part of a much bigger project, I'll try to give you a glance of the code you will find in this folder.

Going folder by folder:
 - `agents`: folder containing classes responsible to map from game state to action (thus here, the class will contain your networks/policies, and implement the logic to map from game state, to network input, and from network output, to environment action)
 - `algorithms`: folder containing classes responsible for the actual training, which will get the data at the end of each game, store it in a buffer, and then every N games, will improve the agent
 - `environment`: folder containing everything about the environment, potentially not really of your interest, if not to formalize the state of the network
 - `main`: folder that contains ready to use code to do the actual training
 - `networks`: folder containing the networks that your agent (in the `agents` folder) will use
 - `scripts`: folder containing scripts for training and evaluation

Most of the code will be already provided, thus you are asked only to complete some classes (depending on weather you go for a value function algorithm or a policy based algorithm)

Specifically you are asked to complete the following files:
 - the network: thus `networks/acnets` for actor critic type of algorithms or `networks/dqnet` for value based algorithms
 - the algorithm: thus `algorithms/a2c` for actor critic, `algorithms/dqn` for value based, specifically you should implement only the `learn` function.   
   **Pay attention** that the "mask" tensor is necessary for the algorithm to converge, and it refers that at that specific state, only certain action were available, thus only those one should be considered for the learning
 
Eventually, you can also play around with the state representation in the `agents` folder, however please pay attention to set the first element of your state to 1 if the current player is the one that that specific agent is referring to (this keep the `turn_state[0] = player.id == current_player.id`)
