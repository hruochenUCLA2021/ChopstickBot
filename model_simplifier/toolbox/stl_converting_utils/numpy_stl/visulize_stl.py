from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot

import os


from stl import mesh ,Mode

# Get the directory of the current script
script_dir = os.path.dirname(__file__)

# Define the path to the 'inputs' folder
input_path = os.path.join(script_dir, 'inputs')
output_path = os.path.join(script_dir, 'outputs')

# Loop through all files in the folder
for file_name in os.listdir(input_path):
    # Check if the file has a .STL extension (case insensitive)
    if file_name.lower().endswith('.stl'):
        file_path_input = os.path.join(input_path, file_name)
        file_path_output = os.path.join(output_path, file_name)


        # Create a new plot
        figure = pyplot.figure()
        axes = figure.add_subplot(projection='3d')

        # Load the STL files and add the vectors to the plot
        '''
        change show input or output
        '''
        file_path = file_path_input
        file_path = file_path_output



        your_mesh = mesh.Mesh.from_file(file_path)
        axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors))

        # Auto scale to the mesh size
        scale = your_mesh.points.flatten()
        axes.auto_scale_xyz(scale, scale, scale)

        # Show the plot to the screen
        pyplot.show(block = False)

pyplot.show(block = True)
