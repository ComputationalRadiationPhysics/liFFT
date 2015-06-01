from helpModule import *
from formFactor import *
from matplotlib import cm
import numpy as np
import sys, getopt

delta_x = 1.
delta_y = 1.

def writeSpectral(data, outFp):
  Nx = len(data[0])
  Ny = len(data)
  kx_max = np.pi * (Nx-1)/(Nx*delta_x)
  ky_max = np.pi * (Ny-1)/(Nx*delta_y)
  sizeX = 1024*20/Nx
  sizeY = 1024*16/Ny
  FIG = plt.figure(0, figsize=(sizeX, sizeY))
  create_2Dmap(data,
               figure=FIG,
               aspect='auto',
               cmap=cm.spectral,
               logScale=logOutput,
               cbar=True,
               xlabel=r"$k_x$",
               ylabel=r"$k_y$",
               extent=(-kx_max, kx_max, -ky_max, ky_max))
  FIG.savefig(outFp, format='pdf')
  plt.close(FIG)
  
def writeNormal(data, outFp):
  Nx = len(data[0])
  Ny = len(data)
  sizeX = 1024*20/Nx
  sizeY = 1024*16/Ny
  FIG = plt.figure(0, figsize=(sizeX, sizeY))
  create_2Dmap(data,
               figure=FIG,
               aspect='auto',
               cmap=cm.spectral,
               logScale=logOutput,
               cbar=True,
               xlabel=r"xPos in mm",
               ylabel=r"yPos in mm")
  FIG.savefig(outFp, format='pdf')
  plt.close(FIG)

def loadAndWriteData(inFp, outFp, isSpectral):
  data = np.loadtxt(inFp, dtype='float32')
  if isSpectral:
    writeSpectral(data, outFp)
  else:
    writeNormal(data, outFp)

  
def main(name, argv):
  inputfile = ''
  outputfile = 'output.pdf'
  isSpectral = False
  try:
    opts, args = getopt.getopt(argv,"hsi:o:",["isSpectral","ifile=","ofile="])
  except getopt.GetoptError:
    print name+' -i <inputfile> -o <outputfile>'
    sys.exit(2)
  for opt, arg in opts:
    if opt == '-h':
       print name+' -i <inputfile> -o <outputfile>'
       sys.exit()
    elif opt in ("-i", "--ifile"):
       inputfile = arg
    elif opt in ("-o", "--ofile"):
       outputfile = arg
    elif opt in ("-s", "--isSpectral"):
       isSpectral = True
  
  print 'Loading "' + inputfile + '" to "' + outputfile + '"'
  loadAndWriteData(inputfile, outputfile, isSpectral)
       
if __name__ == "__main__":
   main(sys.argv[0], sys.argv[1:])
