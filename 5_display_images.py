import cv2
import os
import random
import numpy as np

def get_image_size():
    img = cv2.imread('data/0/100.jpg', 0)
    return img.shape

data = os.listdir('data/')
data.sort(key=int)
begin_index = 0
end_index = 5
image_x, image_y = get_image_size()

if len(data) % 5 != 0:
    rows = int(len(data) / 5) + 1
else:
    rows = int(len(data) / 5)

full_img = None
for i in range(rows):
    col_img = None
    for j in range(begin_index, min(end_index, len(data))):  # Make sure j is within the valid range
        img_path = "data/%s/%d.jpg" % (data[j], random.randint(1, 99))
        print(img_path)  # Debugging: print the img_path
        img = cv2.imread(img_path, 0)

        if img is None:
            img = np.zeros((image_y, image_x), dtype=np.uint8)
        else:
            # Resize image to match the height of the first image in the row
            img = cv2.resize(img, (image_x, image_y))

        if col_img is None:
            col_img = img
        else:
            col_img = np.hstack((col_img, img))

    begin_index += 5
    end_index += 5

    if full_img is None:
        full_img = col_img
    else:
        full_img = np.vstack((full_img, col_img))

cv2.imshow("data", full_img)
cv2.imwrite('full_img.jpg', full_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
