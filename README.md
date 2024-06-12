# Dual Pathway RL 
This is a directory to store checkpoint codes and other info.

---
## Code Info and issues addressed
### 0 Torch REINFORCE: 
- An RL model working in 1D continuous action space using policy gradient REINFORCE method to output a mean $\mu$ and a standard deviation $\sigma$ to generate and action and learn a reward scape.
- Biologically not plausible 
### 1 Single Pathway 1D
- 1D continuous space RL model which is biologically plausible as it works on the basis of reward modulated Hebbian learning. 
- It has 4 components: HVC, BG, RA. MC as shown below
![](images/20240607095346.png)

### 2 Dual Pathway 1D
- In this code, we make an extra connection between HVC and RA as shown below. This connection learns using plain Hebbian learning. 
- This pathway is used to solidify the action which occurs the most number of times making the network robust. 
![](images/20240607095611.png)
### 3 Single Pathway 2D 2 layer
- This is a trial code to expand our learning to 2 dimensions. 
- The faced an issue with directly implementing the whole model, thus I made this trial version to work with.
- This has HVC, BG, MC only with BG being channelized such that X and Y direction are independent of each other. 
### 4 Single pathway 2D 3 layer
- Now adding one more layer i.e. RA layer to the model caused issues such as dependence between x and y output coordinates.
- A scatter plot of this output looked biased in a direction leading to improper learning. The cluster plot of this output is shown below. 
![](images/20240607101712.png)
- In order to overcome this issue, I had to channelized BG, RA such that X and Y motor outputs have no dependence shared between them. The following diagram gives a good idea of the structure of the model. 
![](images/20240607102701.png)
- This solved the issue and made the X and Y independent as shown in the figure below. 
![](images/20240607102834.png)
### 5 Dual Pathway 2D model 
- Now I implemented the Hebbian learning pathway to the previous model i.e. adding a connection between HVC and RA. 
- This adds the robustness to the model as we expect in a song of a songbird. 

### 9 Tuning
#### Influence of BG
Should be more than influence of direct hebbian learning
#### Tuning of Sigmoidal slopes
- we tune the slopes such that the sigmoid(input) should have a good enough range from -1 to 1 without a skew towards the borders.
- We find the slopes of BG_sigmoid to be 2.5 as shown below. 
![](Pasted%20image%2020240612170025.png) 

- Similarly we tune RA slope. Now RA is known to be a bursting kind of neuron, thus a skew in output of RA is physiologically relevant, but we also need to make sure output of MC neuron is not skewed despite of skewed inputs from RA. This is done using the slope of RA sigmoid as 18.
![](Pasted%20image%2020240612165926.png) 
