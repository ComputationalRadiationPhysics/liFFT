"""
Form factor calculation
=======================

This is a script for the calculation of the form factor. As Input you could
use an 2D-array with electron densities of free electrons and bound electrons.

This script is optimize to work as a standalone application or to be started
inside ipython (or other python shell).


developed at HZDR by:

Anton Helm, Lingen Huang, Richard Pausch, Klaus Steiniger and Malte Zacharias
"""

from helpModule import *
from matplotlib import cm
from matplotlib import colors as colors
from matplotlib import pyplot as plt
import numpy as np


#-------------------- PHYSICAL CONSTANTS --------------------------------------
f_0 = 1                     # factor for scattering -- free electrons
f_1 = 1                     # factor for scattering -- bound electrons
f_c = 100                   # factor for complex phase -- resonant
r_e = 2.818e-9              # classical electron radius [micron]

#-------------------- SIMULATION SETUP ----------------------------------------
Nx = 1800                   # cells in x-direction          [-]
Ny = 5400                   # cells in y-direction          [-]
delta_x = 1.
delta_y = 1.
res = 200                   # cells per wave length         [-]
lamd_0 = 0.8                # simulated wave length         [micron]
nc = 1.74e21                # critical density              [cm^-3]

# limits of simulation box
limits = (Nx / res * lamd_0, Ny / res * lamd_0)
xlim, ylim = limits


#-------------------- ION DENSITY DATA ----------------------------------------
ionsFilenames= {
        # charge state      # density data filename
        "+1"            :   "data/Cu_a7_den_ch1_30.dat",
        "+2"            :   "data/Cu_a7_den_ch2_30.dat",
        "+3"            :   "data/Cu_a7_den_ch3_30.dat",
        "+4"            :   "data/Cu_a7_den_ch4_30.dat",
        "+5"            :   "data/Cu_a7_den_ch5_30.dat",
        "+6"            :   "data/Cu_a7_den_ch6_30.dat",
        "+7"            :   "data/Cu_a7_den_ch7_30.dat",
        "+8"            :   "data/Cu_a7_den_ch8_30.dat",
        "+9"            :   "data/Cu_a7_den_ch9_30.dat",
        "+10"           :   "data/Cu_a7_den_ch10_30.dat",
        "+11"           :   "data/Cu_a7_den_ch11_30.dat",
        "+12"           :   "data/Cu_a7_den_ch12_30.dat",
        "+13"           :   "data/Cu_a7_den_ch13_30.dat",
        "+14"           :   "data/Cu_a7_den_ch14_30.dat",
        "+15"           :   "data/Cu_a7_den_ch15_30.dat",
        "+16"           :   "data/Cu_a7_den_ch16_30.dat",
        "+17"           :   "data/Cu_a7_den_ch17_30.dat",
        "+18"           :   "data/Cu_a7_den_ch18_30.dat",
        "+19"           :   "data/Cu_a7_den_ch19_30.dat",
        "+20"           :   "data/Cu_a7_den_ch20_30.dat",
        "+21"           :   "data/Cu_a7_den_ch21_30.dat",
        "+22"           :   "data/Cu_a7_den_ch22_30.dat",
        "+23"           :   "data/Cu_a7_den_ch23_30.dat",
        "+24"           :   "data/Cu_a7_den_ch24_30.dat",
        "+25"           :   "data/Cu_a7_den_ch25_30.dat",
        "+26"           :   "data/Cu_a7_den_ch26_30.dat",
        "+27"           :   "data/Cu_a7_den_ch27_30.dat",
        "+28"           :   "data/Cu_a7_den_ch28_30.dat",
        "+29"           :   "data/Cu_a7_den_ch29_30.dat"
        }

#-------------------- ELECTRON DENSITY DATA -----------------------------------
elecFilenames_free = {
        # type              # density data filenames
        "free"          :   "data/free_e_a7_den30.dat",
        }

elecFilenames_total = {
        # type              # density data filenames
        "total"         :   "data/total_e_a7_den30.dat"
        }


#-------------------- ANALYSIS PARAMETERS -------------------------------------
# region of interest ... [[xmin, xmax], [ymin, ymax]] in cells
region = [
            [ 388,1412],
            [2188,3212]
        ]                   # (1024,1024) in the center of the sim box
#region = [
#            [460, 960],
#            [1700, 3700]
#         ]
#region = [
            #[0, Nx],
            #[0, Ny]
         #]

z_det  = 1.         # distance target-detector [m]
d_det  = 20.e-6     # detector pixel edge length [m]
I0     = 1.e10      # incident photon number

# cross section of incident photon beam [micron^2]
A0 =    (region[0][1] - region[0][0])\
      * (region[1][1] - region[1][0])\
      * (lamd_0/res)**2

# thickness of 2D sheet [cm]
thick_free  = 6e-4  # 6 micron
thick_bound = 6e-5  # .6 micron
thick_res   = 6e-5  # .6 micron


#-------------------- PLOTTING PARAMETERS -------------------------------------
logOutput = True            # plots are logarithmically scaled


#-------------------- SPECIAL FUNCTIONS ---------------------------------------
def reading_data(**kwargs):
    # get filenames of files to look up
    ions_fn = kwargs.pop('ions_fn', None)
    elec_fn = kwargs.pop('elec_fn', None)
    
    # file initialization
    if ions_fn:
        ionsData = IonDensity(
                       ions_fn,                    # ion distributions filenames
                       (Ny,Nx),                    # shape of input data (rows, columns)
                       special="+16",              # charge state to be treated as resonant
                       subgrid=region,             # interesting part of the input
                       atomicNumber=29 ,           # proton number
                       thick_bound=thick_bound,    # thickness of bound nonresonant electron distribution
                       thick_res=thick_res,        # thickness of resonant electron distribution
                       nc=nc                       # critical density
                       )
    else:
        ionsData = 0
    
    if elec_fn:
        elecData = ElecDensity(
                       elec_fn,                    # electron distributions filenames
                       (Ny,Nx),                    # shape of input data (rows, columns)
                       subgrid=region,             # interesting part of the input
                       thick_free=thick_free,      # thickness of free electron distribution
                       nc=nc                       # critical density
                       )
    else:
        elecData = 0

    return ionsData, elecData

def get_scat_dens(ionsData, elecData):
    # get the density ...
    # ... of bound electrons
    if ionsData:
        elec_bound = ionsData.get_dens()
        elec_res   = ionsData.get_res_dens()
    else:
        elec_bound = 0
        elec_res   = 0

    # ... of free electrons
    if elecData:
        elec_free  = elecData.get_dens()
    else:
        elec_free  = 0
    
    #create_2Dmap(elec_free,  aspect="auto", title="free electrons")
    #create_2Dmap(elec_bound, aspect="auto", title="bound electrons")
    #create_2Dmap(elec_res,   aspect="auto", title="resonant electrons")

    # calculate the - generally complex - density of scatterers (electrons)
    scat_dens  =        f_0 * elec_free\
                 +      f_1 * elec_bound\
                 + 1j * f_c * elec_res
    
    return scat_dens 

def calculate_formFactor(scat_dens):
    # calculate the 2d fft (shift puts q=0 axes in the center of the resulting
    # array)
    unshifted        = np.fft.fft2(scat_dens)
    unshifted       *= (lamd_0*1.e-4/res)**2    # multiply by cell size [cm^2]
    fourierTransform = np.fft.fftshift(unshifted)

    # get the corresponding form factor
    formFactor = abs(fourierTransform)**2

    return formFactor

def convert2current( formFactor ):
    # TODO add documentation here
    constant =   I0/A0\
               * r_e**2\
               * d_det**2/z_det**2\

    return constant*formFactor

def plot_formFactor(formFactor, **kwargs):
    vmin_args = kwargs.pop('vmin', 1e13)
    vmax_args = kwargs.pop('vmax', 1e19)

    kx_max = np.pi * (Nx-1)/(Nx*delta_x)
    ky_max = np.pi * (Ny-1)/(Nx*delta_y)

    xlim_args = kwargs.pop('xlim', (-kx_max, kx_max))
    ylim_args = kwargs.pop('ylim', (-ky_max, ky_max))    

    cmap_args = kwargs.pop('cmap', cm.spectral)

    create_2Dmap( formFactor,
                  aspect='auto',
                  logScale=logOutput,
                  interpolation='spline36',
                  origin='lower',
                  cmap=cmap_args,
                  cbar=True,
#                  vmin=vmin_args,
#                  vmax=vmax_args,
                  xlim=xlim_args,
                  ylim=ylim_args,
                  xlabel=r"$k_x$",
                  ylabel=r"$k_y$",
                  extent=(-kx_max, kx_max, -ky_max, ky_max),
                  **kwargs
                )



#-------------------- RUNNING PROGRAMM ----------------------------------------
# starts script if it is called and not loaded
if __name__ == "__main__":
#    ionsData, elecData = reading_data( elec_fn=elecFilenames_total )
    ionsData, elecData = reading_data( elec_fn=elecFilenames_free, ions_fn=ionsFilenames )
    scat_dens          = get_scat_dens( ionsData, elecData )
    formFactor         = calculate_formFactor( scat_dens )
    print "sum of formFactor: %e" % (formFactor.sum())
    current            = convert2current( formFactor )
    print "sum of current: %e" % (current.sum())
#    plot_formFactor(formFactor, xlim=(-1., 1.), ylim=(-1., 1.))
    plot_formFactor(current, xlim=(-1., 1.), ylim=(-1., 1.))
    
    plt.show()
