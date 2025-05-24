import matplotlib.pyplot as plt
import numpy as np
filename = 'loss.txt'
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
plt.title('Loss before changes')
plt.xlabel('Epoch')
plt.ylabel('Loss')
# plt.yticks(np.arange(0, .002, .0002))
plt.savefig('loss.png')
plt.show()
plt.plot(simms_y, ssims)
plt.yticks(np.arange(.2, .7, .1))
plt.title('SSIM before changes')
plt.xlabel('Epoch')
plt.ylabel('SSIM')
plt.savefig('ssim.png')

plt.show()
plt.plot(simms_y, psnrs)
plt.yticks(np.arange(10, 25, 2))
plt.title('PSNR before changes')
plt.xlabel('Epoch')
plt.ylabel('PSNR')
plt.savefig('psnr.png')

plt.show()


