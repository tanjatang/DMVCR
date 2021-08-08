import os
import random
root = '/phys/ssd/tangxueq/tmp/vcr/vcrimage'
mode = 'answer'
split = 'val'
path = os.path.join(root,mode,split)

files = [x for x in os.listdir(path) if x.split('.')[-1] == 'npy']
print(len(files),"jjjjjjjjjjjjjjjjjjj")
# l = int((0.8*len(files)))
# print(l,"llllllllllllllllllll")
# dele = random.sample(files,l)
# for i in dele:
#     os.remove(os.path.join(path,i))
# print(len(files),"finish")
#
