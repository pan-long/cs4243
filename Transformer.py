import cv2
import csv
import numpy as np
from scipy import ndimage

frame_size = 7199
football_field = cv2.imread('images/football_field.png')
maxHeight, maxWidth = football_field.shape[:2]
mask = np.float32([[2881, 153], [5177, 139], [8398, 893], [26, 949]])
mask_scaled = np.float32([[698, 40], [1273, 40], [2094, 225], [2, 234]])

class Transformer():
    def __init__(self, is_scaled):
        if is_scaled:
            self.mask_points = mask_scaled
        else:
            self.mask_points = mask

        # dimension of new image
        self.dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")

        self.M = cv2.getPerspectiveTransform(self.mask_points, self.dst)

    def transform(self, point):
        M = self.M
        x = int((M[0][0] * point[1] + M[0][1] * point[0] + M[0][2]) / (M[2][0] * point[1] + M[2][1] * point[0] + M[2][2]))
        y = int((M[1][0] * point[1] + M[1][1] * point[0] + M[1][2]) / (M[2][0] * point[1] + M[2][1] * point[0] + M[2][2]))

        return (x, y)

    # def initMarker(self, points):
    #   # positions before warp
    #   self.marker_map = { 'C0': (71, 1153), \
    #                       'R0': (80, 761), 'R1': (80, 1033), 'R2': (95, 1127), 'R3': (54, 1156), 'R4': (65, 1185), 'R5': (61, 1204), 'R6': (56, 1217), 'R7': (69, 1213), 'R8': (67, 1253), 'R9': (75, 1281), 'R10': (92, 1347), \
    #                       'B0': (71, 1409), 'B1': (72, 2016), 'B2': (47, 1051), 'B3': (58, 1117), 'B4': (74, 1139), 'B5': (123, 1156), 'B6': (61, 1177), 'B7': (48, 1198), 'R8': (102, 1353)}


def main():
    fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v')
    video_out = cv2.VideoWriter('output.mp4', fourcc, 24.0, (maxWidth, maxHeight), True)

    map = {}
    files = ['R1', 'B2']
    file_readers = []
    transformer = Transformer(True)

    kern = np.hanning(50)   # a Hanning window with width 50
    kern /= kern.sum()      # normalize the kernel weights to sum to 1

    # smooth player movement
    for i in range(len(files)):
        print('===================== smooth player movement: ', i, ' ===================')
        x_coordinates = []
        y_coordinates = []
        reader = open('track/' + files[i] + '.txt')
        
        for j in range(frame_size):
            line = reader.readline().split(',')
            if len(line) > 2:
                point = transformer.transform((int(line[1]), int(line[2])))
                y_coordinates.append(int(point[0]))
                x_coordinates.append(int(point[1]))

        x_coordinates = ndimage.convolve1d(x_coordinates, kern, 0)
        y_coordinates = ndimage.convolve1d(y_coordinates, kern, 0)

        file_name = 'track/' + files[i] + '_smoothed.txt'
        # clear file
        open(file_name, 'w').close()

        with open(file_name, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',)
            for j in range(frame_size):
                writer.writerow([y_coordinates[j], x_coordinates[j]])

    # open file readers
    for f in files:
        file_readers.append(open('track/' + f + '_smoothed.txt'))

    for i in range(frame_size):
        print('===================== process frame: ',  i, ' ===========================')

        red_player_x = []
        blue_player_x = []
        field = np.array(football_field, np.uint8)
        for i in range(len(file_readers)):
            reader = file_readers[i]
            line = reader.readline().split(',')
            point = (int(line[0]), int(line[1]))
            font = cv2.FONT_HERSHEY_SIMPLEX
            if files[i][0] == "B":
                blue_player_x.append(point[0])
                cv2.putText(field, files[i], point, font, 1, (255, 0, 0), 2, cv2.CV_AA)
            else:
                red_player_x.append(point[0])
                cv2.putText(field, files[i], point, font, 1, (0, 0, 255), 2, cv2.CV_AA)
        
        red_max_x = np.amax(red_player_x)
        blue_min_x = np.amax(blue_player_x)
        cv2.line(field, (red_max_x, 0), (red_max_x, maxHeight), (0, 0, 255), 2) 
        cv2.line(field, (blue_min_x, 0), (blue_min_x, maxHeight), (0, 0, 255), 2)

        # cv2.imshow('result', field)
        # cv2.waitKey(1)
        
        video_out.write(field)


if __name__ == '__main__':
    main()