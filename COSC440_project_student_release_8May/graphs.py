import matplotlib.pyplot as plt
import numpy as np

filename = 'loss.txt'

f = open(filename, 'r')
lines = f.readlines()
losses1 = []
ssims1 = []
psnrs1 = []
for line in lines:
    if not(line.startswith('Epoch')):
        continue
    line = line.split()
    if line[2] == 'loss:':
        losses1.append(float(line[3]))
    else:
        ssims1.append(float(line[3]))
        psnrs1.append(float(line[5]))
loss_y1= range(len(losses1))
ssims_y1= range(0, len(ssims1)*10, 10)

filename2 = 'loss_w_attn.txt'
f = open(filename2, 'r')
lines = f.readlines()
losses2 = []
ssims2 = []
psnrs2 = []
for line in lines:
    if not(line.startswith('Epoch')):
        continue
    line = line.split()
    if line[2] == 'loss:':
        losses2.append(float(line[3]))
    else:
        ssims2.append(float(line[3]))
        psnrs2.append(float(line[5]))
loss_y2= range(len(losses2))
ssims_y2= range(0, len(ssims2)*10, 10)

plt.plot(loss_y1, losses1, label='No attention', color='red')
plt.plot(loss_y2, losses2, label='Attention', color='blue')
plt.title('MSE Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss.png')
plt.show()
plt.plot(ssims_y1, ssims1, label='No attention', color='red')
plt.plot(ssims_y2, ssims2, label='Attention', color='blue')
# plt.yticks(np.arange(.2, .7, .1))
plt.title('SSIM')
plt.xlabel('Epoch')
plt.ylabel('SSIM')
plt.legend()
# plt.savefig('ssim.png')

plt.show()
plt.plot(ssims_y1, psnrs1, label='No attention', color='red')
plt.plot(ssims_y2, psnrs2, label='Attention', color='blue')
# plt.yticks(np.arange(10, 25, 2))
plt.title('PSNR')
plt.xlabel('Epoch')
plt.ylabel('PSNR')
plt.legend()
# plt.savefig('psnr.png')

plt.show()


