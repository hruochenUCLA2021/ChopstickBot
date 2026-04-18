import os
import argparse

import open3d as o3d


parser = argparse.ArgumentParser()

parser.add_argument('file', help='STL file to convert')
parser.add_argument("-o", '--output_dir', help='Output directory', required=False)
args = parser.parse_args()

if not os.path.exists(args.file):
    raise "Path does not exist!"

output_dir = os.path.dirname(args.file)
if args.output_dir is not None and os.path.exists(args.output_dir):
    output_dir = args.output_dir

newfile = os.path.join(output_dir, os.path.basename(args.file)[:-4]+".obj")

print(f"Processing {args.file:30} ->    {newfile}")
mesh = o3d.io.read_triangle_mesh(args.file)
o3d.io.write_triangle_mesh(newfile, mesh)
