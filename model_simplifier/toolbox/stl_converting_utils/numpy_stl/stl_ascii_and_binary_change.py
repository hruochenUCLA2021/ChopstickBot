import os
'''
this is using numpy-stl
'''
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
        print("find::",file_name)
        # print("find::",file_name,"in:",file_path_input)

        '''
        change type::
        '''
        this_mesh = mesh.Mesh.from_file(file_path_input)
        # this_mesh.save(file_path_output,mode=Mode.AUTOMATIC)
        # this_mesh.save(file_path_output,mode=Mode.ASCII)
        this_mesh.save(file_path_output,mode=Mode.BINARY)