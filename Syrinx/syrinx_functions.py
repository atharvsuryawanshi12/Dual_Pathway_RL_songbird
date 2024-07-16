#Libraries

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import queue
from scipy.integrate import solve_ivp
from scipy.signal import spectrogram
from scipy.interpolate import interp2d
import warnings
from tqdm import tqdm

np.random.seed(100)

# Parameters
gamma = 12000 
duration = 0.050 #s
dt = 1/44100 #s  #* 0.1 #1.e-4
Amp = 4
nt = int(duration/dt)

# Trachea parameters
r = -0.9
v = 350*100
L = 1.9
tau_0 = 2*L/v # Propagation time along trachea
tau_n = int(tau_0/dt)

syllable = 1
# RC4
# syllable 1
if syllable == 1:
    def Tension(t, T_beta):
        """ Generalised exponential waveform """
        return   .3 - .2*np.exp(-200*t) -.00001 * np.exp(200*t) + T_beta
        
    def Pressure(t, P_alpha):
        """ Constant """
        return 0.04*np.sin((duration/2 + t)*np.pi*10) + 0.05 + P_alpha  
# RC5 - test
elif syllable == 2:
    # syllable 2
    def Tension(t, T_beta):
        """ Generalised exponential waveform """
        Tc = 1#2/3
        return   .3 - .2*np.exp(-Tc*3*100*t) -.00001 * np.exp(Tc*3*100*t) + T_beta  

    def Pressure(t, P_alpha):
        """ Constant """
        tp = 0.2
        return tp*np.sin((duration/2 + t)*np.pi*10) + 0.05 + P_alpha    
elif syllable == 3:
    # syllable 3
    def Tension(t, T_beta): 
        """
            Tension1 = 0.1*np.sin(sin_t*np.pi*40) + 0.6
            Tension2 = 0.2*np.sin(sin_t*np.pi*15) + 0.6
            Tension = np.concatenate((Tension1[:int(np.ceil(nt/2))], Tension2[:int(nt-nt/2)]))
        """
        scale = 1
        if isinstance(t, float):
            if t <= duration*scale/2: 
                return 5*np.sin(t*np.pi*40/scale) + 0.6 + T_beta# old amp 0.1 
            elif t <= duration*scale: 
                return 10*np.sin((t-duration*scale/2)*np.pi*15/scale) + 0.6 + T_beta# old amp 0.2
            else: 
                return 0.0 + 0*t
        elif isinstance(t, np.ndarray):
            Tension1 = 5*np.sin(t*np.pi*40/scale) + 0.6 + T_beta # old amp 0.1
            Tension2 = 10*np.sin(t*np.pi*15/scale) + 0.6 + T_beta # old amp 0.2
            Tension3 = 0.0 + 0*t
            return np.concatenate((Tension1[:int(np.ceil(t.size*scale/2))], Tension2[:int(t.size*scale-t.size*scale/2)], Tension3[:int(t.size-t.size*scale)]))        
    
    def Pressure(t, P_alpha):
        """ Pressure = 0.02 * np.ones((nt)) """
        return np.ones((np.asarray(t).shape)) * 2 + P_alpha # old amp 0.16

elif syllable == 4:
    # syllable 4
    def Tension(t, T_beta):
        scale = 0.4
        if isinstance(t, float):
            if t <= duration*scale: 
                return 0.8*np.sin(t*np.pi*20/scale) + 0.6 + T_beta# old amp 0.1
            else: 
                return t*0 + 0.6 + T_beta# old amp 0.2
        elif isinstance(t, np.ndarray):
            Tension1 = 0.8*np.sin(t*np.pi*20/scale) + 0.6 + T_beta# old amp 0.1
            Tension2 = t* 0 + 0.6 + T_beta# old amp 0.2
            return np.concatenate((Tension1[:int(np.ceil(t.size*scale))], Tension2[:int(t.size-t.size*scale)]))
          
    def Pressure(t, P_alpha):
        """ Pressure = 0.02 * np.ones((nt)) """
        return np.ones((np.asarray(t).shape)) * 0.01 + P_alpha  
        
elif syllable == 5:
    # syllable 5
    def Tension(t, T_beta):
        """ Tension = 0.2 * np.ones((nt)) """
        return np.ones((np.asarray(t).shape)) * 0.2 + T_beta
        
    def Pressure(t, P_alpha):
        """ Pressure = 0.1 * np.ones((nt)) """
        return np.ones((np.asarray(t).shape)) * 0.1 + P_alpha 
elif syllable == 6:
    # syllable 6
    def Tension(t, T_beta):
        """
            Tension1 = 0.1*np.sin(sin_t*np.pi*40) + 0.6
            Tension2 = 0.2*np.sin(sin_t*np.pi*15) + 0.6
            Tension = np.concatenate((Tension1[:int(np.ceil(nt/2))], Tension2[:int(nt-nt/2)]))
        """
        if isinstance(t, float):
            if t <= duration/2: return 0.1*np.sin(t*np.pi*40) + 0.6 + T_beta
            else: return 0.2*np.sin((t-duration/2)*np.pi*15) + 0.6 + T_beta
        elif isinstance(t, np.ndarray):
            Tension1 = 0.1*np.sin(t*np.pi*40) + 0.6 + T_beta
            Tension2 = 0.2*np.sin(t*np.pi*15) + 0.6 + T_beta
            return np.concatenate((Tension1[:int(np.ceil(t.size/2))], Tension2[:int(t.size-t.size/2)]))
        
    def Pressure(t):
        """ Pressure = 0.01*np.sin(sin_t*np.pi*2*100) + 0.02 """
        return 0.01*np.sin(t*np.pi*2*100) + 0.02 + P_alpha 

def syrinxODE(t, y):
    ''' ODEs used in Amador paper'''
    global P_alpha, T_beta
    y0, y1 = y
    dydt = [y1,
         -Pressure(t, P_alpha)*(gamma**2) - Tension(t, T_beta)*(gamma**2)*y0 - (gamma**2)*(y0**3) - gamma*(y0**2)*y1 + (gamma**2)*(y0**2) - gamma*y0*y1
           ]
    return dydt

def Syrinx(P_alpha, T_beta):
    """
    Simulating a syrinx using pressure and tension inputs.

    Args:
        P_alpha: Pressure parameter.
        T_beta: Tension parameter.

    Returns:
        The simulated song.
    """
    # Pre-compute constants outside the loop for efficiency
    if not hasattr(Syrinx, 'Amp'):  # Check if Amp is already defined
        Syrinx.Amp = Amp  # Avoid redundant calculations
    time_x = np.linspace(0, duration, nt)
    # Vectorize pressure calculation (assuming Pressure is vectorized)
    pressure = Pressure(time_x, P_alpha)
    solution = solve_ivp(syrinxODE, [0, duration], [1, 1], method="RK45", t_eval=time_x)
    X = solution['y'][0]
    Y = solution['y'][1]
    song = Syrinx.Amp * pressure * Y
    return song

def Trachea(song):
    """ Simulating the progression of pressure through a trachea. """
    y0 = song.T
    # To generate pressure output from trachea
    P_i=np.zeros((len(y0)))
    # See Fig 1 in Amador paper. Round-about way to implement that.
    Buffer = queue.Queue()
    for i in np.arange(tau_n):
        Buffer.put(np.random.random())
    for i in np.arange(len(y0)):
        P_i[i] = y0[i] - r*Buffer.get();      
        # update_buffer
        Buffer.put(P_i[i])
    # P_tr from P_in (ref Amador paper)
    BufferB=np.zeros((tau_n))
    P_tmp= np.concatenate((BufferB, P_i))
    P_t=(1-r)*P_tmp[:-tau_n]
    return P_t

def plot_gradient(ax, Z, figure):
    """ Plots reward contour. """
    contour = ax.contourf(Z, 25, extent=[0.0, 1.0, 0.0, 0.2], cmap="gray_r", alpha=.25)
#     cbar = figure.colorbar(contour)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.5)
    cbar = figure.colorbar(contour, cax=cax)
    cbar.set_label('Performance metric (R)', rotation=270, fontsize=20, labelpad=25)
    
def create_template(P_alpha=0.02, T_beta=0.6):
    """ Creates spectrogram template from target song. """
    song = Syrinx(P_alpha, T_beta)
    P_tr = Trachea(song)
    (freqs, t, spectrum) = spectrogram(P_tr, fs=1/dt)
    target_spectrum = spectrum

    templateSpec = target_spectrum
    mts = np.mean(templateSpec)
    templateSpec = templateSpec - mts
    templateLen = np.sqrt(np.sum(templateSpec ** 2))
    templateSpec = templateSpec / templateLen
    
    return templateSpec

def compute_corr_coeff(currentSpec, templateSpec):
    """ Computes correlation coefficient. """
    m = np.mean(currentSpec)
    currentSpec = currentSpec - m
    currentLen = np.sqrt(np.sum(currentSpec ** 2))
    currentSpec = currentSpec / currentLen
    
    return np.mean(currentSpec.T@templateSpec)
    
# def generate_gradient(templateSpec, n=256):
#     """ Generates the reward contour by simulating the song for each input combination
#         and storing the similarity metric w.r.t. the target song. """
#     Spectrums = np.zeros((n, n, 129, 9)) # Hardcoded in a hurry
#     # Spectrums = np.load('Figures/spectrums_n'+str(n)+'.npy')
#     Z = np.zeros((n,n))
#     for i in tqdm(np.arange(n)):
#         for j in np.arange(n):
#             P_alpha = i/n*0.2 #+ 0.0
#             T_beta = j/n*1. #+ 0.0
#             song = Syrinx(P_alpha, T_beta)
#             P_tr = Trachea(song)
#             (freqs, t, spectrum) = spectrogram(P_tr, fs=1/dt)
#             Spectrums[i, j] = spectrum
# #             spectrum = Spectrums[i, j]
#             Z[i, j] = compute_corr_coeff(spectrum, templateSpec)
# #     np.save('Figures/spectrums_n'+str(n), Spectrums)
#     return Z

def generate_gradient(templateSpec, P_init, T_init, n=256):
    """ Generates the reward contour by simulating the song for each input combination
        and storing the similarity metric w.r.t. the target song. """
    global P_alpha, T_beta
    delta_P = 0.1
    delta_T = 0.5
    P_range = np.linspace(P_init - delta_P, P_init + delta_P, n)
    T_range = np.linspace(T_init - delta_T, T_init + delta_T, n)
    Spectrums = np.zeros((n, n, 129, 9)) # Hardcoded in a hurry
    # Spectrums = np.load('Figures/spectrums_n'+str(n)+'.npy')
    Z = np.zeros((n,n))
    for i, P_alpha in tqdm(enumerate(P_range)):
        for j, T_beta in enumerate(T_range):    
            song = Syrinx(P_alpha, T_beta)
            P_tr = Trachea(song)
            (freqs, t, spectrum) = spectrogram(P_tr, fs=1/dt)
            Spectrums[i, j] = spectrum
#             spectrum = Spectrums[i, j]
            Z[i, j] = compute_corr_coeff(spectrum, templateSpec)
#     np.save('Figures/spectrums_n'+str(n), Spectrums)
    return Z

# Assign target song
P_init = 0.05
T_init = 0.3

P_alpha = P_init
T_beta = T_init 

templateSpec = create_template(P_alpha, T_beta)
# Generate reward contour based on similarity to assigned target song

nZ = 10 # Specify resolution of performance landscape
Z = generate_gradient(templateSpec, P_init, T_init, nZ)
Z = Z/np.max(Z)
figure = plt.figure(figsize=(8,8))
ax = plt.subplot(frameon=False)

# im = ax.imshow(Z , vmin=0, vmax=Z.max(), cmap='Purples', extent=[0, 0.1, 0, 0.02], aspect='auto')
im = ax.imshow(Z , vmin=Z.min(), vmax=Z.max(), cmap='Purples', extent=[0, 1, 0, 1], aspect='auto')
ax.invert_yaxis() 

# Display colorbar
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.5)
cbar = figure.colorbar(im, cax=cax)
cbar.set_label('Performance metric (R)', rotation=270, fontsize=20, labelpad=25)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()

