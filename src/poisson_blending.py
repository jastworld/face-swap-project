import cv2
import numpy as np
import scipy as sc
import scipy.sparse.linalg


class PoissonBlend:
    def __init__(self):
        print('poisson blend two images')

    def blend (self, new_face, target):
        #--------------- define object_mask  -------------------
        mask_in = np.zeros(new_face[:, :, 0].shape)
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

        # --------------  Poisson Blend Two Images  ----------------
        im_blend = np.zeros(target.shape)
        new_face = cv2.cvtColor(new_face, cv2.COLOR_BGR2RGB).astype('double') / 255.0
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB).astype('double') / 255.0

        for channel in np.arrange(3):
            object_img = new_face[:, :, channel].copy()
            bg_img = target[:, :, channel].copy()

            [ys, xs] = np.nonzero(new_mask)
            yx_list = list(zip(ys, xs))
            yx_dict = dict(zip(yx_list, range(len(yx_list))))

            M = 4 * len(ys)
            N = len(ys)
            A = sc.sparse.lil_matrix((M, N), dtype='double')
            b = np.zeros(M, dtype='double')

            e = 0
            for index in yx_list:
                yn = [index[0] + 1, index[0] - 1, index[0], index[0]]
                xn = [index[1], index[1], index[1] + 1, index[1] - 1]
                for n in range(4):
                    A[e, yx_dict[index]] = 1
                    if (new_mask[yn[n], xn[n]]):
                        A[e, yx_dict[yn[n], xn[n]]] = -1
                    e += 1
            A = sc.sparse.csr_matrix(A)

            e = 0
            for index in yx_list:
                yn = [index[0] + 1, index[0] - 1, index[0], index[0]]
                xn = [index[1], index[1], index[1] + 1, index[1] - 1]

                for n in range(4):
                    if (new_mask[yn[n], xn[n]]):
                        b[e] = object_img[index[0]][index[1]] - object_img[yn[n]][xn[n]]
                    else:
                        b[e] = object_img[index[0]][index[1]] - object_img[yn[n]][xn[n]] + bg_img[yn[n]][xn[n]]
                    e += 1

            v = sc.sparse.linalg.lsqr(A, b)[0]

            vs = np.zeros(bg_img.shape)
            for key, val in yx_dict.items():
                vs[key[0]][key[1]] = v[val]

            im_blend[:, :, channel] = vs * new_mask + bg_img * (1 - new_mask)

        return im_blend
