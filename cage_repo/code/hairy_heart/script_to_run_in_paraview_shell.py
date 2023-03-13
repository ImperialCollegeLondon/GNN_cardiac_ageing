import numpy as np
from paraview.simple import *
from tqdm import tqdm

loop_coordinates_file_baseline='loop_coordinates_any_healthy.txt'
loop_coordinates_file='loop_coordinates_any_unhealthy.txt'
nframes = 50

script_code = (
"""import math
import numpy as np
from vtk.numpy_interface import algorithms as algs
from vtk.numpy_interface import dataset_adapter as dsa
from scipy.special import expit

v = {vtx} # id of the match point to plot

if v==-1:
    print('Special case')
    scalars = np.array([2.25, -2.25])
    coordinates = np.zeros((2,3))
else:
    coordinates = np.loadtxt('{loop_coordinates_file_baseline}')
    coordinates = coordinates.reshape((-1,{nframes},3), order='C')
    coordinates = coordinates[v] # get (nframes, 3) coordinate relative coordinates array
    coordinates[1:] = coordinates[1:] + (coordinates[1:] - coordinates[0]) * 2
    coordinates=np.concatenate((coordinates,coordinates[0,np.newaxis,:]),axis=0) # repeat first line, to get a (nframes, 3) array
    diffs = coordinates - np.concatenate((coordinates[0,np.newaxis,:],coordinates[:-1,:]), axis=0)
    scalars_baseline = np.sum(np.abs(diffs), axis=1)

    coordinates = np.loadtxt('{loop_coordinates_file}')
    coordinates = coordinates.reshape((-1,{nframes},3), order='C')
    coordinates = coordinates[v] # get (nframes, 3) coordinate relative coordinates array
    coordinates[1:] = coordinates[1:] + (coordinates[1:] - coordinates[0]) * 2
    coordinates=np.concatenate((coordinates,coordinates[0,np.newaxis,:]),axis=0) # repeat first line, to get a (nframes, 3) array
    diffs = coordinates - np.concatenate((coordinates[0,np.newaxis,:],coordinates[:-1,:]), axis=0)
    scalars = np.sum(np.abs(diffs), axis=1) - scalars_baseline
    #scalars = expit(scalars)
    #print(min(scalars), max(scalars))

pts = vtk.vtkPoints()
pts.SetData(dsa.numpyTovtkDataArray(coordinates , 'Points'))
output.SetPoints(pts)
numPts=coordinates.shape[0]
index = np.arange(numPts)
output.PointData.append(index , 'Index')
output.PointData.append(scalars , 'Scalars')
ptIds = vtk.vtkIdList()
ptIds.SetNumberOfIds(numPts)
for i in range(numPts): ptIds.SetId(i, i)
output.Allocate (1, 1)
output.InsertNextCell (vtk.VTK_POLY_LINE , ptIds)"""
)


for vtx in tqdm(list(reversed(range(-1, np.loadtxt(loop_coordinates_file).reshape((-1,nframes,3)).shape[0])))):
#for vtx in reversed(range(-1,1)):
    programmableSource1 = ProgrammableSource()
    programmableSource1.Script = script_code.format(
        loop_coordinates_file_baseline=loop_coordinates_file_baseline,
        loop_coordinates_file=loop_coordinates_file,
        vtx=vtx,
        nframes=nframes,
    )
    programmableSource1Display = Show(programmableSource1)
    ColorBy(programmableSource1Display, ('POINTS', 'Scalars'))
    tube1 = Tube(Input=programmableSource1)
    tube1.Radius = 0.7
    tube1Display = Show(tube1)
SaveScreenshot(f'test.png')
