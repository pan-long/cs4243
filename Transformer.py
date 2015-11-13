import cv2
import csv
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt

frame_size = 7199
speed_threshold = 15
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
            [100, 50],
            [maxWidth - 100, 50],
            [maxWidth - 100, maxHeight - 50],
            [100, maxHeight - 50]], dtype = "float32")

        self.M = cv2.getPerspectiveTransform(self.mask_points, self.dst)

    def transform(self, point):
        M = self.M
        x = int((M[0][0] * point[1] + M[0][1] * point[0] + M[0][2]) / (M[2][0] * point[1] + M[2][1] * point[0] + M[2][2]))
        y = int((M[1][0] * point[1] + M[1][1] * point[0] + M[1][2]) / (M[2][0] * point[1] + M[2][1] * point[0] + M[2][2]))

        return (x, y)

    def reverse(self, point):
        M = np.linalg.inv(self.M)
        x = int((M[0][0] * point[0] + M[0][1] * point[1] + M[0][2]) / (M[2][0] * point[0] + M[2][1] * point[1] + M[2][2]))
        y = int((M[1][0] * point[0] + M[1][1] * point[1] + M[1][2]) / (M[2][0] * point[0] + M[2][1] * point[1] + M[2][2]))

        return (x, y)

def main():
    fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v')
    video_out = cv2.VideoWriter('output.mp4', fourcc, 24.0, (maxWidth, maxHeight), True)

    map = {}
    files = ['R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'RE', 'B0', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10']
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
                y_coordinates.append(int(point[1]))
                x_coordinates.append(int(point[0]) - 40)

        x_coordinates = ndimage.convolve1d(x_coordinates, kern, 0)
        y_coordinates = ndimage.convolve1d(y_coordinates, kern, 0)

        file_name = 'track/' + files[i] + '_smoothed.txt'
        # clear file
        open(file_name, 'w').close()
        with open(file_name, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',)
            for j in range(frame_size):
                writer.writerow([x_coordinates[j], y_coordinates[j]])

    # open file readers
    for f in files:
        file_readers.append(open('track/' + f + '_smoothed.txt'))

    prev_x_position = []
    open('track/off_site.txt', 'w').close()
    for i in range(frame_size):
        print('===================== process frame: ',  i, ' ===========================')

        red_player_x = []
        blue_player_x = []
        move_direction = 0
        field = np.array(football_field, np.uint8)

        for i in range(len(file_readers)):
            reader = file_readers[i]
            line = reader.readline().split(',')
            point = (int(line[0]), int(line[1]))
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # update movement direction
            if len(prev_x_position) > i:
                move_direction += (point[0] - prev_x_position[i])
                prev_x_position[i] = point[0]
            else:
                prev_x_position.append(point[0])

            if files[i] == "RE":
                cv2.putText(field, files[i], (point[0]-20, point[1]), font, 1, (0, 0, 0), 2, cv2.CV_AA)
            elif files[i][0] == "B":
                if files[i] != "B0":
                    blue_player_x.append(point[0])
                cv2.putText(field, files[i], (point[0]-20, point[1]), font, 1, (255, 0, 0), 2, cv2.CV_AA)
            else:
                if files[i] != "R0":
                    red_player_x.append(point[0])
                cv2.putText(field, files[i], (point[0]-20, point[1]), font, 1, (0, 0, 255), 2, cv2.CV_AA)
        
        # draw off-site line
        blue_max_x = np.amax(blue_player_x)
        red_min_x = np.amin(red_player_x)
        print move_direction
        if move_direction > speed_threshold:
            top_point = (blue_max_x, 0)
            bottom_point = (blue_max_x, maxHeight)
        elif move_direction < -speed_threshold:
            top_point = (red_min_x, 0)
            bottom_point = (red_min_x, maxHeight)
        else:
            # determine off-site line based on number of players
            left_players = [x for x in (blue_player_x + red_player_x) if x > maxWidth / 2]
            if len(left_players) > len(blue_player_x + red_player_x) / 2:
                top_point = (red_min_x, 0)
                bottom_point = (red_min_x, maxHeight)
            else:
                top_point = (blue_max_x, 0)
                bottom_point = (blue_max_x, maxHeight)
        cv2.line(field, top_point, bottom_point, (0, 0, 255), 2)

        with open('track/off_site.txt', 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',)
            reverse_top = transformer.reverse(top_point)
            reverse_bottom = transformer.reverse(bottom_point)
            writer.writerow([reverse_top[0], reverse_top[1], reverse_bottom[0], reverse_bottom[1]])

        cv2.imshow('result', field)
        cv2.waitKey(1)
        
        # video_out.write(field)


if __name__ == '__main__':
    main()