import matplotlib.pyplot as plt
import numpy as np
filename = 'loss_w_attn.txt'
f = open(filename, 'r')
lines = f.readlines()
losses = []
ssims = []
psnrs = []
for line in lines:
    if not(line.startswith('Epoch')):
        continue
    line = line.split()
    if line[2] == 'loss:':
        losses.append(float(line[3]))
    else:
        ssims.append(float(line[3]))
        psnrs.append(float(line[5]))
loss_y= range(len(losses))
simms_y= range(0, len(ssims)*10, 10)

plt.plot(loss_y, losses)
plt.title(filename + ' losses')
plt.yticks(np.arange(0, .002, .0002))
plt.show()
plt.plot(simms_y, ssims)
plt.yticks(np.arange(0, 1, .1))
plt.title(filename + ' SSIM')
plt.show()
plt.plot(simms_y, psnrs)
plt.yticks(np.arange(10, 30, 2))
plt.title(filename + ' PSNR')
plt.show()


