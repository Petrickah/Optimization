from model import CamereVideo
import numpy as np

camereVideo = CamereVideo()
incapere = camereVideo.get_date()
print("Date intrare:\n", incapere)
print("------------------------------------------------")

from tqdm import tqdm
import time
import math

# solutii = []
# start_time = time.time()
# for i in tqdm(range(1000)):
#     x = camereVideo.gen_nec(False)
#     arie = camereVideo.F(x)
#     solutii.append(np.hstack((x.flatten(), arie)))
# timp = time.time()-start_time
# n_inreg = np.array(solutii).shape[0]
# print("Timp scurs: %.2f secunde / %d înregistrări" % (timp, n_inreg))
# timp_t = (timp*n_inreg)/3600
# print("Timp total pentru 1.000.000 înregistrări: %d ore %d minute" % (math.floor(timp_t), (timp_t-math.floor(timp_t))*60))

# from HillClimbing import HillClimbing

x = camereVideo.gen_nec(False)
print(x.flatten(), '->', camereVideo.F(x))

# hc = Hill(camereVideo, x=x, pas=0.5)
# print("Hill Climbing de bază:")
# optim1 = hc.run_basic()
# print(optim1, '->', hc.valoare(optim1))

# print("---------------------------------------------------")
# print("Hill Climbing cu ascensiune abruptă:")
# optim2 = hc.run_steep(n_iteratii=5)
# print(optim2, '->', hc.valoare(optim2))

# print("---------------------------------------------------")
# print("Hill Climbing cu reporniri aleatoare:")
# optim3 = hc.run_random()
# print(optim3, '->', hc.valoare(optim3))

# from SimulatedAnnealing import SimulatedAnnealing

# x = camereVideo.gen_nec(False)
# print(x.flatten(), '->', camereVideo.F(x))
# sa = SimulatedAnnealing(camereVideo, x=x, pas=0.5)

# x = sa.run()
# print(x, '->', sa.q(x))