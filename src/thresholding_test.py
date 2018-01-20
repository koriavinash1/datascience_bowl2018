import cv2
import sys
from data_loader import DataSet
from config import batch_size

data = DataSet()
X, Y = data.train_batch(batch_size)

image = cv2.imread("../stage1_train/00071198d059ba7f5914a526d124d28e6d010c92466da21d4a04cd5413362552/images/00071198d059ba7f5914a526d124d28e6d010c92466da21d4a04cd5413362552.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh1 = cv2.threshold(image,40, 255, cv2.THRESH_BINARY) # threshold range 12-44

fig = plt.figure()
a = fig.add_subplot(1, 2, 1)
plt.imshow(thresh1, cmap='gray')
a = fig.add_subplot(1, 2, 2)
plt.imshow(image, cmap='gray')

plt.show()