# numpy_ARS
A 150-lines python code for Augmented Random Search (https://arxiv.org/abs/1803.07055) with numpy.

I am definitly impressed by the simplicity and the performance of this algorithm. 
I did not parallelize computations, hence it could be quite less effective than authors' version (https://github.com/modestyachts/ARS).

But my code should be easyer to handle. 

## Quick results (1 episode contains 2*n_directions*horizon environment updates):

### 100 episodes with **HalfCheetah_V1** [step_size=0.02, noise=0.03, n_directions=16, b=16, seed=1]:
![HalfCheetah_V1](img/HalfCheetah_V1.png)

![HalfCheetah_GIF](img/HalfCheetah.gif =250x250)
(took ~10 min on Intel i5 core)

### 100 episodes with **Ant_V1** [step_size=0.015, noise=0.025, n_directions=60, b=20, seed=1]:
![Ant_V1](img/Ant_V1.png)
(took ~1 hour on Intel i5 core)

