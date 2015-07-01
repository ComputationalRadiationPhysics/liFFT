"""
This is a file for providing some helpful functions which are needed inside
the calculation of the form factor.
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import colors as colors

# defines which parts should be available by importing this module
__all__ = ["IonDensity", "ElecDensity", "create_2Dmap", "read_dat", "cmapPerfect"]


# typical field length for data-output
field_length = 16

class SpatialDensity(object):
    """
    Stores the data of a distribution.

    It uses memory-mapping to read the data and stores it as arrays. Beware of
    the fact that it is currently only possible to read string data. The
    strings should have a length of 16 characters and a `EOL`.

    Parameters
    ----------
    densData : str or file-like object
        The file name or file object where the data is stored

    size : tuple, sequence (ny, nx)
        Size of the input with `ny` rows and `nx` columns.


    Optional Parameters
    -------------------
    chargeState : integer
        Highest obtainable charge state for specific species.
        Alias the atomic number.

    special : list of strings
        List which inputs have to be treated in an other way.

    subgrid : list of lists or tuple of tuples
        Specify which area should be read and analyzed.

    """
    def __init__(self, densData, size, **kwargs):
        """
        Constructor of the class `SpatialDensity`
        """
        
        self.data    = {}
        self.grid    = size
        self.charge  = kwargs.pop('atomicNumber', 1)
        self.special = np.array(kwargs.pop('special', ""))
        self.subgrid = kwargs.pop('subgrid', None)
        if 'nc' not in kwargs:
            print("WARNING: NO `nc` GIVEN! DEFAULTS TO 1.")
        self.nc      = kwargs.pop('nc', 1.) # critical density [#electrons/cm^3]

        for key, value in densData.iteritems():
            self.data[key], shape = read_dat( value,
                                              size=size,
                                              area=self.subgrid
                                            )

        #TODO maybe inside method and not outside
        self.shape = shape

    def plot_dens(self, keys, **kwargs):
        """
        Generates plots for specific inputs.

        `kwargs` are passed to the method `create_2Dmap`.

        Parameters
        ----------
        keys : sequence
            Specify which densities should be plotted.

        Optional Parameters
        -------------------
        For optinal parameters go to `create_2Dmap` or to matploblib's
        `imshow`.
        """

        # check if iterable
        if hasattr(keys, '__iter__'):
            for i in keys:
                create_2Dmap(self.data[i], **kwargs)
        # if string is given then it's not iterable
        else:
            create_2Dmap(self.data[keys], **kwargs)

class IonDensity(SpatialDensity):
    """
    Object to store ion densities. It's methods are inherited from
    `SpatialDensity`.

    Additional Methods
    ------------------
    > get_dens(self)
        Return the sum over all non-resonant distributions with the scale of an
        additional scale factor (Z0 - charge state) for every charge state.

    > set_resonant(self, resonant)
        Sets a new resonant charge state.

    > get_res_dens(self)
        Return the sum over all resonant distributions with the scale of an
        additional scale factor (Z0 - charge state) for every charge state.
    """
    def __init__(self, densData, size, **kwargs):
        # TODO add documentation
        super(IonDensity, self).__init__(densData, size, **kwargs)
        if 'thick_bound' not in kwargs:
            print("WARNING: NO `thick_bound` GIVEN. DEFAULTS TO 1.")
        self.thick_bound = kwargs.pop('thick_bound', 1.)
        
        if 'thick_res' not in kwargs:
            print("WARNING: NO `thick_res` GIVEN. DEFAULTS TO 1.")
        self.thick_res   = kwargs.pop('thick_res', 1.)

    def set_resonant(self, resonant):
        # TODO change the documentation at this point
        """
        Sets a new resonant charge state.

        Parameters
        ----------
        resonant : str or list of strings
            Changes which keys have to be treated special.
        """
        self.special = np.array(resonant)
    
    def get_dens(self):
        # TODO change documentation at this point
        """
        Return the sum over all non-resonant distributions with the scale of an
        additional scale factor (Z0 - charge state) for every charge state.
        """
        dens = np.zeros(self.shape)

        for key in self.data.iterkeys():
            dens += (self.charge - int(key)) * self.data[key]

        return self.nc * self.thick_bound * dens

    def get_res_dens(self):
        # TODO change documentation at this point
        """
        Return the sum over all resonant distributions with the scale of an
        additional scale factor (Z0 - charge state) for every charge state.
        """
        dens = np.zeros(self.shape)

        for key in self.data.iterkeys():
            if key in self.special:
                dens += (self.charge - int(key)) * self.data[key]

        return self.nc * self.thick_res * dens


class ElecDensity(SpatialDensity):
    """
    Object to store electron densities. It's methods are inherited from
    `SpatialDensity`.

    Additional Methods
    ------------------
    > get_dens(self)
        Return the sum of all non-special electron densities.

    > set_special(self, special)
        Sets a new special input.

    > get_spec_dens(self)
        Return the sum of all special electron densities.
    """
    def set_special(self, special):
        """
        Sets a new special input.

        Parameters
        ----------
        special : str or list of strings
            Changes which keys have to be treated special.
        """
        self.special = special
    
    def __init__(self, densData, size, **kwargs):
        # TODO add documentation
        super(ElecDensity, self).__init__(densData, size, **kwargs)
        if 'thick_free' not in kwargs:
            print("WARNING: NO `thick_free` GIVEN! DEFAULTS TO 1.")
        self.thick_free = kwargs.pop('thick_free', 1.)
    
    def get_dens(self):
        # TODO change documentation at this point
        """
        Return the sum of all non-special electrons.
        """
        dens = np.zeros(self.shape)

        if self.charge != 1:
            print("CHARGE OF THE ELECTRONS SHOULD BE ONE!!!!")
            self.charge = 1

        for key in self.data.iterkeys():
            if not key in self.special:
                dens += self.charge * self.data[key]
                break

        return self.nc * self.thick_free * dens
    
    def get_spec_dens(self):
        """
        Return the sum of all special electron densities.
        """
        dens = np.zeros(self.shape)

        if self.charge != 1:
            print("CHARGE OF THE ELECTRONS SHOULD BE ONE!!!!")
            self.charge = 1

        for key in self.data.iterkeys():
            if key in self.special:
                dens += self.charge * self.data[key]
                break

        print("`get_spec_dens()` returns density without multiplying by any thickness value!")
        return self.nc * dens

def read_dat(fp, size, area=None):
    """
    Reads a text file with the memmory mapping method and returns the a 2d
    array with strings. It's just to read a part of the file.

    Uses the view method to restruct the array of strings and convert the array
    to a float array.

    Parameters
    ----------
    fp : str or file object
        Specification which file should be read.

    size : tuple, sequence (ny, nx)
        Size of the input with `ny` rows and `nx` columns.

    Optional Parameters
    -------------------
    area : list of lists or tuple of tuples
        Specify which area should be read.
    """

    print(">> reading file ..."+fp)
    # dtype with all coloumns as field names and EOL
    memtype = create_dtype(size)
    fields = list(memtype.names)
    field_type = "S"+str(field_length)

    # return array without EOL
    dat = np.memmap(fp, dtype=memtype, mode='r')[fields[:-1]]
    # transform view into array
    dat = dat.view(dtype=field_type).reshape(size)
    if area is not None:
        left, right = area[0]
        lower, upper = area[1]
        dat = dat[lower:upper, left:right]

    return np.float32(dat), np.shape(dat)


def create_dtype(size):
    """
    Creates a dtype of string fields which is needed for the buffer inside the
    memory mapping method of numpy.

    Parameters
    ----------
    size : tuple, sequence (ny, nx)
        Size of the input with `ny` rows and `nx` columns.
    """
    Ny, Nx = size

    dtype_str = "S"+str(field_length)+","
    dtype_str += ("S" + str(field_length) + ",")*(Nx - 1)
    dtype_str += "S1"

    return np.dtype(dtype_str)


def create_2Dmap(dat,  **kwargs):
    """
    Creates a 2D plot of any data and tunnel any plot specific parameters to
    the imshow method.

    Parameters
    ----------
    dat : 2d array
        2d array which should be plotted.
    """
    # getting all interesting plot options
    xlimits = kwargs.pop('xlim', None)
    ylimits = kwargs.pop('ylim', None)
    xlabel = kwargs.pop('xlabel', None)
    ylabel = kwargs.pop('ylabel', None)
    labelfontsize = kwargs.pop('labelfontsize', 20)
    titlefontsize = kwargs.pop('titlefontsize', 24)
    tickfontsize = kwargs.pop('tickfontsize', 16)
    logPlot = kwargs.pop('logScale', None)
    title = kwargs.pop('title', None)
    cbar = kwargs.pop('cbar', None)
    figure = kwargs.pop('figure', None)
    subplot = kwargs.pop('subplot', None)

    if figure == None:
        figure = plt.figure()

    if subplot == None:
        subplot = figure.add_subplot(111)



    if logPlot:
        pic = subplot.imshow(dat, norm=LogNorm(), **kwargs)
    else:
        pic = subplot.imshow(dat, norm=None, **kwargs)

    if xlimits:
        plt.xlim(xlimits)
    if ylimits:
        plt.ylim(ylimits)

    if xlabel:
        plt.xlabel(xlabel, fontsize=labelfontsize)
    if ylabel:
        plt.ylabel(ylabel, fontsize=labelfontsize)

    plt.xticks(size=tickfontsize)
    plt.yticks(size=tickfontsize)

    if title:
        plt.title(title, fontsize=titlefontsize)
    if cbar:
        CB = figure.colorbar(pic)
        for t in CB.ax.get_yticklabels():
            t.set_fontsize(tickfontsize)
        





# create my color map
cdict = {'red': ((0.0, 1, 1),
                 (0.03, 0, 0),
                 (0.35, 0, 0),
                 (0.66, 1, 1),
                 (0.89,1, 1),
                 (1, 0.5, 0.5)),
         'green': ((0.0, 1, 1),
                   (0.03, 0, 0),
                   (0.125,0, 0),
                   (0.375,1, 1),
                   (0.64,1, 1),
                   (0.91,0,0),
                   (1, 0, 0)),
         'blue': ((0.0, 1, 1),
                  (0.03, 0.8, 0.8),
                  (0.11, 1, 1),
                  (0.34, 1, 1),
                  (0.65,0, 0),
                  (1, 0, 0))
         }

cmapPerfect = colors.LinearSegmentedColormap('TheColormap',cdict,256)

