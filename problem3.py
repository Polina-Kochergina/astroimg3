from matplotlib import pyplot as plt
from astropy.io import fits
from matplotlib.colors import LogNorm
import numpy as np
from scipy.fftpack import fft2, fftshift, ifft2
from PIL import Image

 
original_img = Image.open('06510002.JPG')
arr = np.asarray(original_img, dtype='uint8')
x, y, _ = arr.shape
 
k = np.array([[[0.2989, 0.587, 0.114]]])
arr2 = np.round(np.sum(arr*k, axis=2)).astype(np.uint8).reshape((x, y))
 
img2 = Image.fromarray(arr2)
img2.save('result_bw.png')  #перевели  в чб

fourier_img = fft2(img2) # Forward transform
print(fourier_img.shape)
size = 2048
print(size)

image = []
image_array = []

print(abs(fourier_img).mean(), abs(fourier_img.min()))
q = [0, 0.1, 0.5, 0.7, 0.8, 0.9, 0.99, 0.995, 0.999, 0.9999]  # доля выкинутых коэф
qua = np.quantile(abs(fourier_img), q)
max = qua[-2]; min = qua[-1]
q = [1, 1-0.1, 1-0.5, 1-0.7, 1-0.8, 1-0.9, 1-0.99, 1-0.995, 1-0.999, 1-0.9999]  # доля оставленных коэф



print(min, max)
koef = []
for i in range(10):                                                     # зануляеи все что меньше порога
    # fourier_img[fourier_img <= abs(max - min) * i/10] = 0
    fourier_img[abs(fourier_img) < abs(qua[i])] = 0

    image_array = np.append(image_array, fourier_img)



image_array = image_array.reshape(10,2048,3089)                         # решейпим массив


# считаем коэф разности изобр от кол-ва коэф фурье
for i in range(10):
    k = np.linalg.norm(abs(image_array[i,:,:]) - abs(image_array[0,:,:]))/np.linalg.norm(abs(image_array[0,:,:]))
    koef.append(abs(k))
    print(k)

# fig, axes = plt.subplots(5, 2, figsize=(12, 12))
# график
plt.plot(q, koef)
plt.ylabel("delta")
plt.xlabel("share coefficient")

# сохраняеи картинки
# plt.semilogy()
# for i in range(10):
#     # for j in range(2):
#     img = ifft2(image_array[i,:,:])
#     print(img.shape)
#     # axes[i][j].imshow(abs(img), cmap = 'gray', norm = LogNorm())
#     plt.imshow(abs(img), cmap = 'gray')
#     plt.savefig(f"image_{i}.png")


plt.show()