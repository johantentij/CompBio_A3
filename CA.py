import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.sparse as sp

N = 100
dx = 1
dt = .1

x = np.arange(N) * dx

### nutrient diffusion
D = 1
initNutrient = 10

# 2d laplacian with Neumann BCs
D2 = sp.diags((np.ones(N - 1), -2 * np.ones(N), np.ones(N)), (-1, 0, 1)).tolil()

D2[0, 1] = 2
D2[N-1, N-2] = 2
D2 = D2.tocsr()

I = sp.eye(N, format="csr")

D2_2 = sp.kron(I, D2) + sp.kron(D2, I)

nutrients = initNutrient * np.ones(N ** 2, dtype=np.float64)

def updateNutrients(nutrients):
    nutrients += dt * D * D2_2.dot(nutrients)

    nutrients[nutrients < 0] = 0

    return


### bacteria properties
def monod(x, K):
    return x / (x + K)

# how much nutrient consumed per timestep
nutrientConsumption = .1

# how much nutrient for reproduction
K_reproduction = 50

bacteria_indices = [N ** 2 // 2 + N // 2]
bacteria = np.zeros(N ** 2, dtype=np.bool_)
bacteria[bacteria_indices[0]] = True

# allowed movement neighbourhood
neighbourhood = np.array([
    0,
    -N, 
    N,
    -1,
    -N-1,
    N-1,
    1,
    -N+1,
    -N-1
])


### simulation function

def simulationStep():
    # diffuse nutrient
    updateNutrients(nutrients)

    deathNote = []
    for i in range(len(bacteria_indices)):
        b_pos = bacteria_indices[i]

        # dinner time
        nutrients[b_pos] -= nutrientConsumption

        # die if not enough food
        if (nutrients[b_pos] <= 0):
            nutrients[b_pos] = 0

            bacteria[b_pos] = False
            deathNote.append(i)
            
            continue
        
        region = neighbourhood + b_pos

        # don't move outside domain
        isValid = (region >= 0) & (region < N ** 2)
        region = region[isValid]

        # prevent wrapping
        x = b_pos % N
        xRegion = region % N
        region = region[np.abs(x - xRegion) < 5]
        

        # don't move on top of other bacteria
        notOccupied = ~bacteria[region]

        p_reproduction = 0.01 * monod(nutrients[b_pos], nutrientConsumption)
        
        # check if reproduce
        if (np.random.random() < p_reproduction):
            region = region[notOccupied]

            if (np.shape(region)[0] > 0):
                newPos = np.random.choice(region, 1)
                bacteria_indices.append(newPos)
                bacteria[newPos] = True

        # otherwise move to more food
        else:
            # include probability of remaining stationary
            notOccupied[0] = True
            region = region[notOccupied]

            if (np.shape(region)[0] > 0):
                p = nutrients[region]
                p /= np.sum(p)

                newPos = np.random.choice(region, 1, p=p)
                bacteria[bacteria_indices[i]] = False
                bacteria_indices[i] = newPos
                bacteria[newPos] = True

    # execute death note, start from highest to avoid problems
    for i in reversed(deathNote):
        bacteria_indices.pop(i)

    return

fig, ax = plt.subplots()

nutrients_2D = np.reshape(nutrients, (N, N), order='F')
bacteria_2D = np.reshape(bacteria, (N, N), order='F')
plot = np.copy(nutrients_2D)
plot[bacteria_2D] = np.nan
im = ax.imshow(plot, vmin=0, vmax=initNutrient, cmap="winter")

def update(frame):
    simulationStep()

    nutrients_2D = np.reshape(nutrients, (N, N), order='F')
    bacteria_2D = np.reshape(bacteria, (N, N), order='F')
    plot = np.copy(nutrients_2D)
    plot[bacteria_2D] = np.nan
    im.set_data(plot)
    
    return [im]
    
ani = animation.FuncAnimation(
    fig, update, interval=100, cache_frame_data=False
) 

plt.show()   
