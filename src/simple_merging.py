import cv2
import numpy as np

class SimpleMerge:
    def __init__(self):
        print('simple merge two images')

    def merge(self, new_face, target):
        #--------------- define object_mask  -------------------
        mask_in = np.zeros(new_face.shape)
        for b in np.arange(3):
            mask_in += new_face[:, :, b]
        mask_in[mask_in != 0] = 1

        [ys, xs] = np.nonzero(mask_in)
        yx_list = list(zip(ys, xs))

        mask_temp = mask_in
        new_mask = mask_in
        for index in yx_list:
            y = index[0]
            x = index[1]
            flag = mask_temp[y + 1, x] * mask_temp[y - 1, x] * mask_temp[y, x + 1] * mask_temp[y, x - 1]
            new_mask[index] = flag

        # --------------  Simple Merge Two Images  ----------------
        im_merge = np.zeros(target.shape)
        new_face = cv2.cvtColor(new_face, cv2.COLOR_BGR2RGB).astype('double') / 255.0
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB).astype('double') / 255.0

        for channel in np.arrange(3):
            object_img = new_face[:, :, channel].copy()
            bg_img = target[:, :, channel].copy()
            im_merge[:, :, channel] = object_img * new_mask + bg_img * (1-new_mask)

        return im_merge
