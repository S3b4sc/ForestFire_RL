from matplotlib.colors import ListedColormap
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


# Functions for animations
colors = ['green', 'red', 'black', 'white']
ticksLabels = ['No tree', 'Healthy tree', 'Burning tree', 'Burned tree']
customCmap = ListedColormap(colors)
ticksLocation = [0.375, 1.125, 1.875, 2.625]

def squareAnimationPlot(filename:str, historical:list, interval:int, p_bond, p_site) -> None:

    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.matshow(historical[0], cmap=customCmap, vmin=1, vmax=4)
    ax.set_title('Square tessellation simulation', size=20)
    ax.set_xlabel(r'$P_{bond}=$' + str(round(p_bond,2)) + r'  $P_{site}=$' + str(round(p_site,2)), size=15)
    cbar = plt.colorbar(cax, ticks=ticksLocation)
    cbar.set_ticklabels(ticksLabels)
    cbar.set_label('Tree status')

    # Función de actualización de la animación
    def update(i):
        cax.set_array(historical[i])  # Actualizar la matriz mostrada
        return [cax]

    # Configuración de la animación
    squareAni = animation.FuncAnimation(
        fig, update, frames=len(historical), interval=interval, blit=True)
    # Mostrar la animación
    squareAni.save(filename + ".gif", writer="pillow")
    
# Parameters for training and testing PPO agent
parameters = {
'neighbours': [(-1,0),(1,0),(0,1),(0,-1)],
'neighboursBoolTensor': np.ones((4,15,15), dtype=bool),
'grid_size':15,
'threshold':0.55,
'max_steps': 70,
'extinguish_limit':1
}