import os
import shutil

dir = "/home/isec/Documents/new_tokenized_slice"

dir1 = "/home/isec/Documents/tokenizeds_slice_1"
dir2 = "/home/isec/Documents/tokenizeds_slice_2"
i = 0
for root, dirs,files in os.walk(dir):
  for file in files:
    if i %2 == 0:
      shutil.move(os.path.join(root, file), os.path.join(dir1, file))
      print(file)
    else:
      shutil.move(os.path.join(root, file), os.path.join(dir2, file))
      print(file)
    i += 1
  