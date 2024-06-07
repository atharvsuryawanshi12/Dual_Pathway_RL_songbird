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
![[Pasted image 20240607095346.png]]

### 2 Dual Pathway 1D
- In this code, we make an extra connection between HVC and RA as shown below. This connection learns using plain Hebbian learning. 
- This pathway is used to solidify the action which occurs the most number of times making the network robust. 
![[Pasted image 20240607095611.png]]
### 3 Single Pathway 2D 2 layer
- This is a trial code to expand our learning to 2 dimensions. 
- The faced an issue with directly implementing the whole model, thus I made this trial version to work with.
- This has HVC, BG, MC only with BG being channelized such that X and Y direction are independent of each other. 
