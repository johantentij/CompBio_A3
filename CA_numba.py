import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.sparse as sp

from numba import njit

N_x = 300
N_y = 600
dx = 1
dt = .1

x = np.arange(N_x) * dx
y = np.arange(N_y) * dx

### nutrient diffusion
D = 1
initNutrient = 10

nutrients = initNutrient * np.ones((N_x, N_y), dtype=np.float64)

nutrientSource = 1e-1
# degradation; set equilibrium at initNutrient
nutrientDegradation = nutrientSource / initNutrient

@njit
def dNutrients_dt(nutrients):
    diffusion = np.empty((N_x, N_y), dtype=np.float64)

    # y = 0
    diffusion[0, 0] = 2 * (nutrients[1, 0] + nutrients[0, 1]) - 4 * nutrients[0, 0]
    for x in range(1, N_x - 1):
        diffusion[x, 0] = nutrients[x - 1, 0] + nutrients[x + 1, 0] + 2 * nutrients[x, 1] - 4 * nutrients[x, 0]
    diffusion[N_x - 1, 0] = 2 * (nutrients[N_x - 2, 0] + nutrients[N_x - 1, 1]) - 4 * nutrients[N_x - 1, 0]

    # 1 < y < N_y - 1 (Interior rows)
    for y in range(1, N_y - 1):
        # Left edge (x = 0)
        diffusion[0, y] = 2 * nutrients[1, y] + nutrients[0, y-1] + nutrients[0, y+1] - 4 * nutrients[0, y]
        
        # Interior points
        for x in range(1, N_x - 1):
            diffusion[x, y] = (nutrients[x-1, y] + nutrients[x+1, y] + 
                               nutrients[x, y-1] + nutrients[x, y+1] - 4 * nutrients[x, y])
        
        # Right edge (x = N_x - 1)
        diffusion[N_x - 1, y] = 2 * nutrients[N_x - 2, y] + nutrients[N_x - 1, y-1] + nutrients[N_x - 1, y+1] - 4 * nutrients[N_x - 1, y]

    # y = N_y - 1 (Bottom boundary)
    diffusion[0, N_y - 1] = 2 * (nutrients[1, N_y - 1] + nutrients[0, N_y - 2]) - 4 * nutrients[0, N_y - 1]
    for x in range(1, N_x - 1):
        diffusion[x, N_y - 1] = nutrients[x-1, N_y - 1] + nutrients[x+1, N_y - 1] + 2 * nutrients[x, N_y - 2] - 4 * nutrients[x, N_y - 1]
    diffusion[N_x - 1, N_y - 1] = 2 * (nutrients[N_x - 2, N_y - 1] + nutrients[N_x - 1, N_y - 2]) - 4 * nutrients[N_x - 1, N_y - 1]
            
    return D * diffusion + nutrientSource - nutrients * nutrientDegradation

def updateNutrients(nutrients):
    # modified euler
    k1 = dNutrients_dt(nutrients)
    k2 = dNutrients_dt(nutrients + k1 * dt)

    nutrients += .5 * dt * (k1 + k2)

    nutrients[nutrients < 0] = 0

    return nutrients


### bacteria properties
def monod(x, K):
    return x / (x + K)

# how much nutrient consumed per timestep
nutrientConsumption = .1

# how much nutrient for reproduction
K_reproduction = 5
n_reproduction = 3

bacteria_indices = np.empty((N_x * N_y, 2), dtype=np.int32)
bacteria_indices[0] = [N_x // 2, 0]
aliveCount = 1

bacteria = np.zeros((N_x, N_y), dtype=np.bool_)
bacteria[bacteria_indices[0, 0], bacteria_indices[0, 1]] = True

# probability of chemotaxis
p_chemotaxis = 1e-3


### antibiotics
steps = 6
baseConcentration = 1
antibiotic = np.empty((N_x, N_y), dtype=np.float64)
for i in range(steps):
    antibiotic[:, i * N_y // steps:] = baseConcentration * i + 1e-2

### genes
p_mutation = 2e-5
mutation_step = 1
K_antibiotic = np.empty(N_x * N_y, dtype=np.float64) 
n_antiobiotic = 3

# eve's gene
K_antibiotic[0] = 1.5

genomeIDs = np.empty((N_x * N_y, 4), dtype=np.int32)
genomeCounter = 0
# genome ID: [parent, self, firstX, firstY]
genomeIDs[0, :] = [genomeCounter, genomeCounter, bacteria_indices[0, 0], bacteria_indices[0, 1]]


### simulation function

@njit
def bacteriaStep(nutrients, bacteria, bacteria_indices, antibiotic, K_antibiotic, aliveCount, genomeIDs, genomeCounter):
    # allowed movement neighbourhood
    neighbourhood = [
        [0, 1],
        [0, -1],
        [1, 0],
        [1, 1],
        [1, -1],
        [-1, 0],
        [-1, 1],
        [-1, -1]
    ]

    i = 0
    while i < aliveCount:
        b_pos = bacteria_indices[i]

        # die if not enough food or from antibiotic
        x = antibiotic[b_pos[0], b_pos[1]]
        den = x ** n_antiobiotic + K_antibiotic[i] ** n_antiobiotic

        if den == 0:
            p_death = 0.0
        else:
            p_death = 0.1 * x ** n_antiobiotic / den

        if (nutrients[b_pos[0], b_pos[1]] <= 0 or 
            np.random.random() < p_death):

            bacteria[b_pos[0], b_pos[1]] = False
            
            bacteria_indices[i] = bacteria_indices[aliveCount - 1]
            K_antibiotic[i] = K_antibiotic[aliveCount - 1]
            genomeIDs[i] = genomeIDs[aliveCount - 1]
            aliveCount -= 1


            if aliveCount == 0:
                return aliveCount, genomeCounter

            # rerun for this index, since it has been changed to different bacterium
            # (no i += 1)
            
            continue
        
        nutrientsLocal = nutrients[b_pos[0], b_pos[1]]
        p_reproduction = 0.05 * nutrientsLocal ** n_reproduction / (nutrientsLocal ** n_reproduction + K_reproduction ** n_reproduction)

        # check if reproduce
        if (np.random.random() < p_reproduction):
            allowedRegion = np.zeros((9, 2), dtype=np.int32)
            count = 0
            for offset in neighbourhood:
                newX = b_pos[0] + offset[0]
                newY = b_pos[1] + offset[1]

                # check if newX, newY is valid point
                if (newX >= 0 and newX < N_x 
                    and newY >= 0 and newY < N_y):

                    # check if occupied
                    if (not bacteria[newX, newY]):
                        allowedRegion[count] = [newX, newY]
                        count += 1

            if (count > 0):
                x_random = count * np.random.random()

                pSum = 0
                for j in range(count):
                    pSum += 1
                    if (pSum > x_random):
                        newX, newY = allowedRegion[j]
                        break

                # set new bacteria
                bacteria[newX, newY]= True
                bacteria_indices[aliveCount, :] = [newX, newY]

                K_base = K_antibiotic[i]
                genomeID = genomeIDs[i]
                # check if mutation
                if (np.random.random() < p_mutation):
                    # check direction
                    if (np.random.random() < .5):
                        K_base += mutation_step
                    else:
                        K_base -= mutation_step

                    K_base = max(0, K_base)

                    genomeCounter += 1
                    genomeID[0] = genomeID[1]
                    genomeID[1] = genomeCounter
                    genomeID[2] = newX
                    genomeID[3] = newY
                    
                K_antibiotic[aliveCount] = K_base
                genomeIDs[aliveCount] = genomeID
                aliveCount += 1

        # otherwise move to more nutrient
        elif (np.random.random() < p_chemotaxis):
            allowedRegion = np.zeros((9, 2), dtype=np.int32)
            allowedRegion[0] = b_pos
            count = 1

            totNutrient = nutrients[b_pos[0], b_pos[1]]
            for offset in neighbourhood:
                newX = b_pos[0] + offset[0]
                newY = b_pos[1] + offset[1]

                # check if newX, newY is valid point
                if (newX >= 0 and newX < N_x 
                    and newY >= 0 and newY < N_y):

                    # check if occupied, excluding self
                    if (not bacteria[newX, newY]):
                        allowedRegion[count] = [newX, newY]
                        count += 1

                        totNutrient += nutrients[newX, newY]

            if (count > 0):
                x_random = totNutrient * np.random.random()

                pSum = 0
                for j in range(count):
                    x, y = allowedRegion[j]
                    pSum += nutrients[x, y]
                    if (pSum > x_random):
                        newX, newY = allowedRegion[j]
                        break         

                # move bacteria
                bacteria[b_pos[0], b_pos[1]] = False
                bacteria[newX, newY] = True
                bacteria_indices[i] = [newX, newY]

        i += 1

    return aliveCount, genomeCounter



### plot full CA
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

plot_nutrients = np.copy(nutrients)
plot_nutrients[bacteria] = np.nan
im1 = ax1.imshow(plot_nutrients, vmin=0, vmax=initNutrient, cmap="winter", origin='lower')
ax1.set_title("Nutrient Concentration")
plt.colorbar(im1, ax=ax1, label="Nutrients")

gene_map = np.full((N_x, N_y), np.nan)
gene_map[bacteria_indices[0, 0], bacteria_indices[0, 1]] = K_antibiotic[0]

im2 = ax2.imshow(gene_map, cmap="rainbow", origin='lower', vmin=K_antibiotic[0], vmax=8)
ax2.set_title("Gene: Antibiotic Resistance (K_antibiotic)")
cbar2 = plt.colorbar(im2, ax=ax2, label="Resistance Level")

fig.tight_layout()

frameStep = 50



def update(frame):
    global aliveCount, nutrients, bacteria, genomeCounter

    for _ in range(frameStep):
        # diffuse nutrients
        nutrients = updateNutrients(nutrients)

        # dinner time
        nutrients[bacteria] -= nutrientConsumption

        # increment CA
        aliveCount, genomeCounter = bacteriaStep(nutrients, bacteria, bacteria_indices, antibiotic, K_antibiotic, aliveCount, genomeIDs, genomeCounter)

    plot_n = np.copy(nutrients)
    plot_n[bacteria] = np.nan
    im1.set_data(plot_n)

    # make gene plot
    current_gene_map = np.full((N_x, N_y), np.nan)
    
    if aliveCount > 0:
        active_indices = bacteria_indices[:aliveCount]
        active_genes = K_antibiotic[:aliveCount]
        
        for idx in range(aliveCount):
            gx, gy = active_indices[idx]
            current_gene_map[gx, gy] = active_genes[idx]
    
    im2.set_data(current_gene_map)
    
    return [im1, im2]

ani = animation.FuncAnimation(
    fig, update, interval=1, cache_frame_data=False, blit=False
) 

plt.show()

### plot history map
# downscale = 5
# N_x_red = N_x // downscale
# N_y_red = N_y // downscale

# history = np.empty((N_x_red, N_y_red), dtype=np.float64)
# history[:, :] = np.nan
# historyTime = 0

# frameStep = 50
# timeCycle = 2000
# def updateHistory():
#     global historyTime

#     for x in range(N_x_red):
#         for y in range(N_y_red):
#             hasBacteria = np.sum(bacteria[x * downscale:(x+1) * downscale, y * downscale: (y+1) * downscale]) > 0

#             if hasBacteria and np.isnan(history[x, y]):
#                 history[x, y] = historyTime % timeCycle
#                 historyTime += 1

#     return

# genomeTree = [genomeIDs[0, :]]
# genomeLines = []

# substantialThreshold = 50

# def updateGenomeTree():
#     activeGenomes = genomeIDs[:aliveCount]
#     selfIDs = activeGenomes[:, 1]
#     _, uniqueIndices, uniqueCounts = np.unique(selfIDs, return_index=True, return_counts=True)

#     uniqueIndices = uniqueIndices[uniqueCounts > substantialThreshold]

#     genomes = activeGenomes[uniqueIndices]

#     for genome in genomes:
#         self = genome[1]

#         isNew = True
#         for genomeComp in genomeTree:
#             selfComp = genomeComp[1]

#             if (self == selfComp):
#                 isNew = False

#                 break

#         if isNew:
#             # find parent
#             parent = genome[0]

#             for genomeComp in genomeTree:
#                 selfComp = genomeComp[1]

#                 if (parent == selfComp):
#                     xParent = genomeComp[2] // downscale
#                     yParent = genomeComp[3] // downscale

#                     xChild = genome[2] // downscale
#                     yChild = genome[3] // downscale

#                     genomeLines.append([[xParent, xChild], [yParent, yChild]])

#                     break

#             genomeTree.append(genome)


# updateHistory()

# fig, ax = plt.subplots()

# im = ax.imshow(history, vmin=0, vmax=timeCycle, cmap="hsv")

# def update(frame):
#     global nutrients, bacteria, aliveCount, genomeCounter

#     for _ in range(frameStep):
#         # diffuse nutrients
#         nutrients = updateNutrients(nutrients)

#         # dinner time
#         nutrients[bacteria] -= nutrientConsumption

#         # increment CA
#         aliveCount, genomeCounter = bacteriaStep(
#             nutrients, 
#             bacteria, bacteria_indices, 
#             antibiotic, K_antibiotic, 
#             aliveCount, 
#             genomeIDs, genomeCounter
#         )

#     updateHistory()
#     updateGenomeTree()

#     for xVals, yVals in genomeLines:
#         ax.plot(yVals, xVals, color="white")

#     im.set_data(history)

#     return [im]

# ani = animation.FuncAnimation(
#     fig, update, interval=1, cache_frame_data=False, blit=False
# ) 

# plt.show()
