import pickle
import numpy as np
import matplotlib.pyplot as plt

# Parameters
nb_pixel_side = 40 #40 #80
TIME_SPENT = 100
rho =0.02 # 100000 #100
name = 'outputs/MAM_normal_b1_4parallel_1000s_M_100_0.02rho.pkl' # f'outputs/MAM_b1_2parallel_{TIME_SPENT}s_M_100_{rho}rho_2.pkl' #MAM_normal_b1_4parallel_1000s_M_100_0.02rho
with open(name, 'rb') as f:
    RES_MAM = pickle.load(f)
plt.figure()
plt.imshow(np.reshape(RES_MAM[0], (nb_pixel_side,nb_pixel_side)) , cmap='hot_r')
plt.colorbar()
plt.show()

# Set up the axes
l_rho = [0.02, 1, 10, 100, 1000, 5000, 10000, 100000]
fig, axs = plt.subplots( 1, len(l_rho), figsize=(20,20)) #(9,5.5)
# Iterate over the plotting locations
i = 0
for ax in axs.ravel():
    try:
        rho = l_rho[i]
        name = f'outputs/MAM_b1_4parallel_{TIME_SPENT}s_M_100_{rho}rho_2.pkl'  # f'outputs/MAM_b1_4parallel_{TIME_SPENT}s_M_100_{rho}rho_{gamma}unbalanced_2.pkl' # f'outputs/local_MAM_4parallel_{TIME_SPENT}s_M_50_{rho}rho_{gamma}unbalanced_2.pkl'
        with open(name, 'rb') as f:  # __{eta} # _centersMNIST #_dataset1
            RES_MAM = pickle.load(f)
        print(f"rho = {rho}, iterations = {RES_MAM[-1]}, weight={np.sum(RES_MAM[0])}")
        im = ax.imshow(np.reshape(RES_MAM[0], (nb_pixel_side,nb_pixel_side)) , cmap='hot_r') #, vmax=0.0008) #10**-3)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title( r"$\rho =$" + f" {rho}" , fontsize = 10.0) #r"$\eta =$" + f" {eta}," +
    except:
        pass
    i = i + 1

# colorbar:
fig.subplots_adjust(right=0.83)
cbar_ax = fig.add_axes([0.85, 0.15, 0.01, 0.7])

plt.show()

# Compare iteration per iteration
name1 = f'outputs/MAM_no_proj_b1_4parallel_100s_M_100_every_it.pkl'
name2 =  f'outputs/MAM_b1_4parallel_100s_M_100_every_it.pkl'
WD = {}
distB = {}
for name in [name1, name2]:
    with open(name, 'rb') as f:  # __{eta} # _centersMNIST #_dataset1
        RES_MAM = pickle.load(f)
    N = RES_MAM[-1]
    WD[name] = RES_MAM[3][:N]
    distB[name] = RES_MAM[4][:N]
    fig, axs = plt.subplots( 5, 10, figsize=(20,20)) #(9,5.5)
    # Iterate over the plotting locations
    i = 0
    for ax in axs.ravel():
        try:
            print(f"rho = {rho}, iterations = {RES_MAM[-1]}, weight={np.sum(RES_MAM[0])}")
            im = ax.imshow(np.reshape(RES_MAM[1][:,i], (nb_pixel_side,nb_pixel_side)) , cmap='hot_r') #, vmax=0.0008) #10**-3)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title( r"k =" + f" {i}" , fontsize = 10.0) # , WD={RES_MAM[3][i]}, distB={RES_MAM[4][i]}
        except:
            pass
        i = i + 1
    # colorbar:
    fig.subplots_adjust(right=0.83)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.01, 0.7])
    # fig.colorbar(cax=cbar_ax)
    plt.show()

N1 = len(WD[name1])
N2 = len(WD[name2])
plt.figure()
plt.plot(np.linspace(1,N1,N1), WD[name1], label='no proj')
plt.plot(np.linspace(1,N2,N2), WD[name2], label='proj')
plt.legend()
plt.grid()
plt.show()
# DistaB
plt.figure()
plt.plot(np.linspace(1,N1,N1), distB[name1], label='no proj')
plt.plot(np.linspace(1,N2,N2), distB[name2], label='proj')
plt.legend()
plt.grid()
plt.show()
