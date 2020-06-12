import cv2
import numpy as np
from matplotlib import pyplot as plt

path = r'vegeta.jpg'  # enter image path here
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
height = np.size(fshift, 0)
width = np.size(fshift, 1)
center_h, center_w = int(height/2), int(width/2)
cutoff = 15
order = 1

H = np.zeros((height, width))
for h in range(height):
	for w in range(width):
		if (((h / height - 0.5) ** 2 + (w / width - 0.5) ** 2) ** 0.5) != 0:
			H[h][w] = 1 / (1 + (cutoff / (((h / height - 0.5) ** 2 + (w / width - 0.5) ** 2) ** 0.5)) ** (2 * order))

output = fshift * H
output = np.fft.ifftshift(output)
output = np.fft.ifft2(output)
output = np.abs(output)

plt.subplot(221),plt.imshow(H, cmap='gray')
plt.title('Butterworth High Filter, cutoff=15'), plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(output, cmap='gray')
plt.title('Output, cutoff=15'), plt.xticks([]), plt.yticks([])


cutoff = 150
H = np.zeros((height, width))
for h in range(height):
	for w in range(width):
		if (((h / height - 0.5) ** 2 + (w / width - 0.5) ** 2) ** 0.5) != 0:
			H[h][w] = 1 / (1 + (cutoff / (((h / height - 0.5) ** 2 + (w / width - 0.5) ** 2) ** 0.5)) ** (2 * order))

output = fshift * H
output = np.fft.ifftshift(output)
output = np.fft.ifft2(output)
output = np.abs(output)


plt.subplot(222),plt.imshow(H, cmap='gray')
plt.title('Butterworth High Pass Filter, cutoff=150'), plt.xticks([]), plt.yticks([])
plt.subplot(224),plt.imshow(output, cmap='gray')
plt.title('Output, cutoff=150'), plt.xticks([]), plt.yticks([])
plt.show()