import torch
from torch.utils.data import Dataset
import numpy as np
import random
import cv2
import math
import matplotlib.pyplot as plt


class NewMetaDataset(Dataset):
    def __init__(self, file_path, k_shot, k_qry, tasks_len=4):
        self.file_path = file_path
        self.k_shot = k_shot
        self.k_query = k_qry
        self.nb_tasks = tasks_len
        self.tolerant = 0
        self.pick_n = 0
        self.threshold = 0.85

    def __len__(self):
        return self.nb_tasks

    def __getitem__(self, idx):
        """
        return: k-shot support images and k-query images in the same task
        """
        task_path = self.file_path + '/task_' + str(idx) + '/'
        self.pick_n = np.loadtxt(task_path + 'length')
        self.pick_n = random.randint(0, self.pick_n - 1)
        support = torch.zeros(self.k_shot, 4, 448, 448)
        query = torch.zeros(self.k_query, 4, 448, 448)
        support_label = torch.zeros(self.k_shot, 9, 28, 28)  # Wait to be modified
        query_label = torch.zeros(self.k_query, 9, 28, 28)  # Wait to be modified

        self.bright = random.randint(0, 10)
        self.b_plus = random.randint(0, 8) + self.bright
        self.g_plus = random.randint(0, 8) + self.bright
        self.r_plus = random.randint(0, 8) + self.bright
        self.pickplace = random.randint(0, 1)
        self.channel_aug = random.randint(0, 5)

        i = j = 0
        lb = torch.zeros(9, 28, 28)
        self.threshold = 0.5
        for n in random.sample(range(0, 41), self.k_shot):
            # self.tolerant = 1
            color_img, depth_img, label = self.get_tensor_data(idx, n)
            lb[0:6] = label[0:6]
            lb[6:9] = label[12:15]
            rgbd_img = torch.zeros(4, 448, 448)
            rgbd_img[0:3] = color_img
            rgbd_img[3] = depth_img
            support[i] = rgbd_img
            support_label[i] = lb
            i += 1
        self.threshold = 0.55
        for n in random.sample(range(0, 41), self.k_query):
            # self.tolerant = 1
            color_img, depth_img, label = self.get_tensor_data(idx, n)
            lb[0:6] = label[0:6]
            lb[6:9] = label[12:15]
            rgbd_img = torch.zeros(4, 448, 448)
            rgbd_img[0:3] = color_img
            rgbd_img[3] = depth_img
            query[j] = rgbd_img

            query_label[j] = lb
            j += 1

        return support, support_label, query, query_label

    def get_tensor_data(self, idx, demo_n):

        # print('---------task-------demo:', idx, demo_n)
        demo_path = self.file_path + '/task_' + str(idx) + '/demo_' + str(demo_n)
        color_img_path = demo_path + '/realsense_color_image.jpg'
        color_img = cv2.imread(color_img_path)
        #         color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
        depth_img = np.load(demo_path + '/realsense_depth_image.npy')
        # depth_img = np.load(demo_path + '/realsense_depth_image.npy')
        # plt.imshow(depth_img)
        # plt.show()

        pick = np.load(demo_path + '/grasp_in_image_' + str(self.pick_n) + '.npy')
        place = np.load(demo_path + '/grasp_in_image_0.npy')
        color_img, depth_img, label = self.data_augmentation(color_img, depth_img, pick, place)

        return color_img, depth_img, label

    def data_augmentation(self, color_img, depth_img, pick, place):
        """
        Realtime data augmentation include rotate/translation/flip/colorJitter
        @param color_img: bgr numpy array (480, 640, 3)
        @param depth_img:   (480, 640)
        @param pick: list [x, y, angle, length]
        @param place:
        return: chw image and label
        """

        # generate image of size (448, 448)
        i = random.randint(0, 10)
        if i < 6:
            color_img, depth_img, pick, place = self.translation_augmentation(color_img, depth_img, pick, place)
        elif i == 6:
            color_img, depth_img, pick, place = self.amplify(color_img, depth_img, pick, place)
        else:
            color_img, depth_img, pick, place = self.rotate_augmentation(color_img, depth_img, pick, place)
        # flip or rotate(big) augmentation
        color_img, depth_img, pick, place = self.flip_rotate(color_img, depth_img, pick, place)
        color_img = self.color_change(color_img)

        label = self.make_grid_label(pick, place, depth_img, scale=28)

        color_img = torch.from_numpy(np.transpose(color_img, (2, 0, 1)))
        depth_img = torch.from_numpy(np.expand_dims(depth_img, axis=0))

        return color_img / 255.0, (depth_img) / 700.0, label

    def make_grid_label(self, pick, place, depth_img, scale=28):
        """
        generate grid cell labels
        @param scale:
        @param pick:
        @param place:
        """
        label = torch.zeros(18, scale, scale)
        # neg1_points = self.find_neg1(depth_img, scale=28)
        pick_pos_points, pick_pos_angles, pick_zero_points = self.find_positives(pick, scale, threshold=self.threshold)
        place_pos_points, place_pos_angles, place_zero_points = self.find_positives(place, scale, threshold=self.threshold)
        if pick[3] < 30:
            pick_length = 1
        elif pick[3] < 50:
            pick_length = 2
        else:
            pick_length = 3

        if place[3] < 30:
            place_length = 1
        elif place[3] < 50:
            place_length = 2
        else:
            place_length = 3

        for pick_neg2 in place_pos_points:
            for i in range(0, 9):
                label[i][pick_neg2[0]][pick_neg2[1]] = -2

        for i in range(18):
            for pz in pick_zero_points:
                if 0 <= pz[0] < 28 and 0 <= pz[1] < 28:
                    label[i, pz[0], pz[1]] = -1
        # set most place with object to -1 , which means they are a little hard samples
        # for p in neg1_points:
        #     for i in range(0, 18):
        #         label[i][p[0]][p[1]] = -1

        # set positive points to 1 and their neighbours to 0
        for angle in pick_pos_angles:
            # print('angle:', angle)

            for pp in pick_pos_points:
                if 0 <= pp[0] < 28 and 0 <= pp[1] < 28:
                    label[pick_length + 11][pp[0]][pp[1]] = 1
                    label[angle][pp[0]][pp[1]] = 1
        #             for pz in pick_zero_points:
        #                 label[angle][pz[0]][pz[1]] = 0

        # set place object to -2, which means they are hard samples
        # for pick_neg2 in place_pos_points:
        #     for i in range(0, 6):
        #         label[i][pick_neg2[0]][pick_neg2[1]] = -2
        #     for i in range(12, 15):
        #         label[i][pick_neg2[0]][pick_neg2[1]] = -2

        # the same the place
        for angle in place_pos_angles:
            # print('angle:', angle)
            for pp in place_pos_points:
                if 0 <= pp[0] < 28 and 0 <= pp[1] < 28:
                    label[place_length + 14][pp[0]][pp[1]] = 1
                    label[angle + 6][pp[0]][pp[1]] = 1
        #             for pz in place_zero_points:
        #                 label[angle + 6][pz[0]][pz[1]] = 0
        # for place_neg2 in pick_pos_points:
        #     for i in range(0, 6):
        #         label[i + 6][place_neg2[0]][place_neg2[1]] = -2
        #     for i in range(15, 18):
        #         label[i][place_neg2[0]][place_neg2[1]] = -2

        # print('positive points:', positive_points)
        # print('positive angles:', positive_angles)
        return label

    def color_change(self, color_img):

        b, g, r = cv2.split(color_img)

        # print(b.shape)
        b += self.b_plus
        g += self.g_plus
        r += self.r_plus
        # print(self.channel_aug)
        # print(self.b_plus)
        b += random.randint(0, 10)
        g += random.randint(0, 10)
        r += random.randint(0, 10)

        if self.channel_aug == 1:
            color_img = cv2.merge([r, g, b])
        elif self.channel_aug == 0:
            color_img = cv2.merge([r, b, g])
        elif self.channel_aug == 2:
            color_img = cv2.merge([b, r, g])
        elif self.channel_aug == 3:
            color_img = cv2.merge([b, g, r])
        elif self.channel_aug == 4:
            color_img = cv2.merge([g, b, r])
        elif self.channel_aug == 5:
            color_img = cv2.merge([g, r, b])

        return color_img

    def find_neg1(self, depth_img, scale=28):

        grid_size = int(448 / scale)
        neg1_points = []
        neighbour = [(-1, -1), (-1, 0), (0, 0), (0, -1)]

        for x in range(1, 28):
            for y in range(1, 28):
                cent_x = x * grid_size - 1
                cent_y = y * grid_size - 1
                if depth_img[cent_x, cent_y] < 0.9143:
                    for nbr in neighbour:
                        neg1_points.append((x + nbr[0], y + nbr[1]))

        return neg1_points

    def find_positives(self, act, scale=28, threshold=0.8):

        positive_points = []
        positive_angle = []
        zero_points = []

        grid_size = 448 / scale

        x = int(act[0] / grid_size)
        y = int(act[1] / grid_size)
        if x != 28 and y != 28:
            positive_points.append((y, x))
        # if self.tolerant:
        #     neighbour = [(1, -1), (1, 0), (1, 1), (0, -1), (0, 1), (-1, -1), (-1, 0), (-1, 1)]
        # else:
        #     neighbour = []

        neighbour = [(1, 0), (0, -1), (0, 1), (-1, 0), (0, 0)]

        for a in neighbour:
            x0 = x + a[0]
            y0 = y + a[1]
            #             print('(x0, y0):', (x0, y0))

            # act presented in image, so x, y have to be the same
            if 0 <= x0 < scale and 0 <= y0 < scale:
                cent_x = x0 * grid_size + grid_size / 2
                cent_y = y0 * grid_size + grid_size / 2
                l = math.sqrt((cent_x - act[0]) ** 2 + (cent_y - act[1]) ** 2)
                #                 print('l:', l)
                if l < threshold * grid_size and y0 < 28 and y0 >=0 and x0 < 28 and x0 >= 0:
                    positive_points.append((y0, x0))
                else:
                    zero_points.append((y0, x0))

        theta = act[2] / math.pi * 180
        m = theta / 30 + 3
        m0 = math.floor(m)
        m1 = math.ceil(m)

        if math.fabs(m - m0) <= threshold:
            positive_angle.append(m0)
        if math.fabs(m - m1) <= threshold:
            if m1 == 6:
                m1 = 0
            positive_angle.append(m1)

        return positive_points, positive_angle, zero_points

    def flip_rotate(self, color_img, depth_img, pick, place):

        # pick as to center
        pick[0] -= 224
        pick[1] -= 224
        place[0] -= 224
        place[1] -= 224

        i = random.randint(1, 8)
        if i <= 3:  # rotate 90 180 270 degree
            theta = - 90 * i / 180 * math.pi
            if i == 1:
                color_img = cv2.rotate(color_img, cv2.ROTATE_90_CLOCKWISE)
                depth_img = cv2.rotate(depth_img, cv2.ROTATE_90_CLOCKWISE)
            elif i == 2:
                color_img = cv2.rotate(color_img, cv2.ROTATE_180)
                depth_img = cv2.rotate(depth_img, cv2.ROTATE_180)
            elif i == 3:
                color_img = cv2.rotate(color_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                depth_img = cv2.rotate(depth_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

            rot_matrix = np.asarray([[math.cos(theta), math.sin(theta)], [-math.sin(theta), math.cos(theta)]])
            a = np.dot(rot_matrix, np.asarray([[pick[0]], [pick[1]]]))
            pick[0] = a[0]
            pick[1] = a[1]
            pick[2] -= theta

            a = np.dot(rot_matrix, np.asarray([[place[0]], [place[1]]]))
            place[0] = a[0]
            place[1] = a[1]
            place[2] -= theta

        elif i == 4:  # horizontal flip
            color_img = cv2.flip(color_img, flipCode=1)
            depth_img = cv2.flip(depth_img, flipCode=1)
            pick[0] = -pick[0]
            pick[2] = -pick[2]
            place[0] = -place[0]
            place[2] = -place[2]
        elif i == 5:  # vertical flip
            color_img = cv2.flip(color_img, flipCode=0)
            depth_img = cv2.flip(depth_img, flipCode=0)
            pick[1] = -pick[1]
            pick[2] = -pick[2]
            place[1] = -place[1]
            place[2] = -place[2]

        if pick[2] > math.pi / 2:
            pick[2] -= math.pi
        if pick[2] >= math.pi / 2:
            pick[2] -= math.pi

        if place[2] > math.pi / 2:
            place[2] -= math.pi
        if place[2] >= math.pi / 2:
            place[2] -= math.pi

        pick[0] += 224
        pick[1] += 224
        place[0] += 224
        place[1] += 224

        return color_img, depth_img, pick, place

    def translation_augmentation(self, color_img, depth_img, pick, place):
        """

        @param color_img:
        @param depth_img:
        @param pick:
        @param place:
        return image of size 448  448
        """
        i = random.randint(8, 40)
        j = random.randint(0, 53)  # relief spatial
        x0 = 67 + j  # bias
        y0 = i - 8
        color_img = color_img[y0: (y0 + 448), x0: (x0 + 448)]
        depth_img = depth_img[y0: (y0 + 448), x0: (x0 + 448)]
        pick[0] -= x0
        pick[1] -= y0
        place[0] -= x0
        place[1] -= y0
        return color_img, depth_img, pick, place

    def amplify(self, color_img, depth_img, pick, place):
        pick[0] -= 320
        pick[1] -= 240
        place[0] -= 320
        place[1] -= 240
        depth_img = cv2.resize(depth_img[42:438, 122:518], (448, 448))
        color_img = cv2.resize(color_img[42:438, 122:518], (448, 448))
        pick[0] *= 1.13
        pick[1] *= 1.13
        pick[3] *= 1.13
        place[0] *= 1.13
        place[1] *= 1.13
        place[3] *= 1.13

        pick[0] += 224
        pick[1] += 224
        place[0] += 224
        place[1] += 224
        return color_img, depth_img, pick, place

    def rotate_augmentation(self, color_img, depth_img, pick, place):
        """

        @param color_img:
        @param depth_img:
        @param pick:
        @param place:
        return image of size 448  448
        """
        theta = random.randint(-20, 20)
        M = cv2.getRotationMatrix2D((320, 240), theta, 1)
        depth_img = cv2.warpAffine(depth_img, M, (640, 480))[16: 464, 96:544]
        color_img = cv2.warpAffine(color_img, M, (640, 480))[16: 464, 96:544]
        pick[0] -= 320
        pick[1] -= 240
        place[0] -= 320
        place[1] -= 240

        theta = theta / 180 * math.pi
        rot_matrix = np.asarray([[math.cos(theta), math.sin(theta)], [-math.sin(theta), math.cos(theta)]])
        a = np.dot(rot_matrix, np.asarray([[pick[0]], [pick[1]]]))
        pick[0] = a[0] + 224
        pick[1] = a[1] + 224
        pick[2] += theta

        a = np.dot(rot_matrix, np.asarray([[place[0]], [place[1]]]))
        place[0] = a[0] + 224
        place[1] = a[1] + 224
        place[2] += theta

        return color_img, depth_img, pick, place


def plot_grid(img):
    cell_size = 14
    (h, w, a) = img.shape
    w_space = int(w / cell_size)
    h_space = int(h / cell_size)
    for i in range(1, cell_size):
        img[i * h_space, :, :] = 0
    for j in range(1, cell_size):
        img[:, j * w_space, :] = 0
    return img


if __name__ == '__main__':
    path = '/home/gwk/URX_GRASP/MousePick/DataSet'
    dataset = MetaDataset(path, 2, 1)
    dataset.__getitem__(81)
