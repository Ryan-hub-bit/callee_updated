import os 
import shutil


binary_dir = "/home/isec/Documents/binary/o1"
new_binary_dir = "/home/isec/Documents/binary/json"

for root, dirs, files in os.walk(binary_dir):
  for file in files:
    json_file = file + ".tgcfi.json"
    if os.path.exists(os.path.join(root,json_file)):
      print(os.path.join(root,json_file))
      shutil.copy2(os.path.join(root,json_file), os.path.join(new_binary_dir, json_file))
    