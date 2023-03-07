import matplotlib.pyplot as plt
import skimage.io
import cv2
import numpy as np

def decrease_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    v = v / 1.0
    v *= value
    v = v.astype(np.uint8)
    # v[v < value] = 0
    # v[v >= value] -= value
    final_hsv = cv2.merge((h,s,v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
    return img

img = skimage.io.imread(f'/home/cindy/PycharmProjects/data/ocr/test/ic_test_fine3/9.png')
img = decrease_brightness(img, value=0.6)
img = img / 255.0
img = img + np.random.randn(*img.shape) * 0.05
img = np.clip(img, 0, 1.0)

plt.figure()
plt.imshow(img)
plt.show()