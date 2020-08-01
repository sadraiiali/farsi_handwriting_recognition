import glob
import os
import random

import cv2
import numpy as np
import six.moves.cPickle as pickle

output_folder = './out/'
dataset_folder = './dataset/'
img_height = 1490
img_width = 1049
max_output_size = 75

RED = '\033[91m'
GREEN = '\033[92m'
BOLD = '\033[1m'

box_size = (1049 / 14, 1490 / 21)
hough_min_votes = 50
hough_distance_resolution = 1
hough_angle_resolution = np.pi / 20
find_line_trh = 10

farsi_digits = [
    '۰', '۱', '۲', '۳', '۴', '۵', '۶', '۷', '۸', '۹',
    'ا', 'ب', 'پ', 'ت', 'ث', 'ج', 'چ', 'ح', 'خ', 'د',
    'ذ', 'ر', 'ز', 'ژ', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ',
    'ع', 'غ', 'ف', 'ق', 'ک', 'گ', 'ل', 'م', 'ن', 'و',
    'ه', 'ی',
]

page_one_digit = [0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 2, 3]
page_two_digit = [4, 5, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 6, 7, 8, 9]

output_dataset = {}

output_dataset['all'] = []

output_dataset['classified'] = []
for i in range(42):
    output_dataset['classified'].append([])

s_id_to_file = {}


def findAruco(img, one_or_two=True):
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters_create()
    marker_corners, marker_ids, rejected_candidates = cv2.aruco.detectMarkers(img, dictionary, parameters=parameters)

    marker_ids = marker_ids.tolist()
    a_30_i = marker_ids.index([30])
    a_31_i = marker_ids.index([31])
    a_33_i = marker_ids.index([33])
    a_32_i = marker_ids.index([32])

    p1 = int(marker_corners[a_30_i][0][0][0]), int(marker_corners[a_30_i][0][0][1])
    p2 = int(marker_corners[a_31_i][0][1][0]), int(marker_corners[a_31_i][0][1][1])
    p3 = int(marker_corners[a_33_i][0][2][0]), int(marker_corners[a_33_i][0][2][1])
    p4 = int(marker_corners[a_32_i][0][3][0]), int(marker_corners[a_32_i][0][3][1])

    return np.array([p1, p2, p3, p4], dtype=np.float32) if one_or_two else np.array([p3, p4, p1, p2], dtype=np.float32)


def correctPerspective(img, aruco_positions, correct_form, output_size):
    H = cv2.getPerspectiveTransform(aruco_positions, correct_form)
    J = cv2.warpPerspective(img, H, output_size)
    return J


def doPerspectiveAndSaveRow(aruco_positions, img, filename):
    x = 5
    n = int(210 * x)
    m = int(297 * x)
    output_size = (n, m)
    correct_form = np.array([
        (0, 0), (n, 0),
        (n, m), (0, m)
    ], dtype=np.float32)

    img = correctPerspective(img, aruco_positions, correct_form, output_size)
    output_filename = out_folder + 'row/' + filename.split('/')[-1]
    cv2.imwrite(output_filename, img)

    return img


def findCorners(img):
    # Now, apply corner detection on Ck
    Ck = np.float32(img)
    window_size = 4
    soble_kernel_size = 5  # kernel size for gradients
    alpha = 0.04
    H = cv2.cornerHarris(Ck, window_size, soble_kernel_size, alpha)
    H = H / H.max()
    C = np.uint8(H > 0.01) * 255
    nC, CC, stats, centroids = cv2.connectedComponentsWithStats(C)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(Ck, np.float32(centroids), (5, 5), (-1, -1), criteria)
    return corners, nC


def findXAndYLines(img):
    G = img
    E = cv2.Canny(G, 80, 100)

    L = cv2.HoughLines(E, hough_distance_resolution, hough_angle_resolution, hough_min_votes)
    all_x_lines = []
    all_y_lines = []
    for [[rho, theta]] in L:
        if -0.01 < theta / np.pi < 0.01:
            all_x_lines.append(np.cos(theta) * rho)

        if 0.49 < theta / np.pi < 0.51:
            all_y_lines.append(np.sin(theta) * rho)

    x_founded = []
    for i in range(1, 14):
        tmp = []
        for j in all_x_lines:
            if i * box_size[0] - find_line_trh < float(j) < i * box_size[0] + find_line_trh:
                tmp.append(int(j))
        x_founded.append(tmp)

    y_founded = []
    for i in range(1, 21):
        tmp = []
        for j in all_y_lines:
            if i * box_size[1] - find_line_trh < float(j) < i * box_size[1] + find_line_trh:
                tmp.append(int(j))
        y_founded.append(tmp)

    output_x_lines = []
    output_y_lines = []
    counter = 0
    for j in x_founded:
        counter += 1
        size = len(j)
        mean = 0
        if size != 0:
            for i in range(len(j)):
                mean += j[i]
            output_x_lines.append(mean / size)
        else:
            output_x_lines.append(int(counter * box_size[0]))
    counter = 0
    for j in y_founded:
        counter += 1
        size = len(j)
        mean = 0
        if size != 0:
            for i in range(len(j)):
                mean += j[i]
            output_y_lines.append(mean / size)
        else:
            output_y_lines.append(int(counter * box_size[1]))
    return output_x_lines, output_y_lines


def saveTile(img, one_or_two, line, id, last_start, last_end, s_id):
    name = ''
    if one_or_two:
        name = page_one_digit[line]
    else:
        name = page_two_digit[line]
    output_file_name = out_folder + 'extracted/{}/{}.png'.format(name, id)

    x_size = int(last_end[0]) - int(last_start[0])
    y_size = int(last_end[1]) - int(last_start[1])

    # x_offset = max_output_size - x_size
    # y_offset = max_output_size - y_size
    # np.ceil(x_offset / 2)
    output_img = img[int(last_start[1]):int(last_end[1]), int(last_start[0]):int(last_end[0])]
    output_img = cv2.resize(output_img, (max_output_size, max_output_size), None, interpolation=cv2.INTER_CUBIC)
    output_img = output_img[5:max_output_size - 5, 5:max_output_size - 5]

    output_dataset['all'].append([name, output_img])
    output_dataset['classified'][int(name)].append(output_img)

    # cv2.imshow('corners', output_img)
    # cv2.waitKey(0)  # press any key
    cv2.imwrite(output_file_name, output_img)


def dump_to_pkl():
    random.shuffle(output_dataset['all'])
    data_size = len(output_dataset['all'])
    data = {}
    data['digits'] = {}
    data['letters'] = {}
    data['digits']['train'] = {}
    data['digits']['train']['data'] = []
    data['digits']['train']['target'] = []
    data['digits']['val'] = {}
    data['digits']['val']['data'] = []
    data['digits']['val']['target'] = []
    data['digits']['test'] = {}
    data['digits']['test']['data'] = []
    data['digits']['test']['target'] = []
    data['letters']['train'] = {}
    data['letters']['train']['data'] = []
    data['letters']['train']['target'] = []
    data['letters']['val'] = {}
    data['letters']['val']['data'] = []
    data['letters']['val']['target'] = []
    data['letters']['test'] = {}
    data['letters']['test']['data'] = []
    data['letters']['test']['target'] = []
    train_size = int((data_size * 80) / 100)
    val_size = int((data_size * 15) / 100)

    for i in range(0, train_size):
        if int(output_dataset['all'][i][0]) < 10:
            data['digits']['train']['data'].append(output_dataset['all'][i][1])
            data['digits']['train']['target'].append(output_dataset['all'][i][0])
        else:
            data['letters']['train']['data'].append(output_dataset['all'][i][1])
            data['letters']['train']['target'].append(output_dataset['all'][i][0])
    for i in range(train_size, train_size + val_size):
        if int(output_dataset['all'][i][0]) < 10:
            data['digits']['val']['data'].append(output_dataset['all'][i][1])
            data['digits']['val']['target'].append(output_dataset['all'][i][0])
        else:
            data['letters']['val']['data'].append(output_dataset['all'][i][1])
            data['letters']['val']['target'].append(output_dataset['all'][i][0])
    for i in range(train_size + val_size, data_size):
        if int(output_dataset['all'][i][0]) < 10:
            data['digits']['test']['data'].append(output_dataset['all'][i][1])
            data['digits']['test']['target'].append(output_dataset['all'][i][0])
        else:
            data['letters']['test']['data'].append(output_dataset['all'][i][1])
            data['letters']['test']['target'].append(output_dataset['all'][i][0])
    print(GREEN, '🗂 -> ', 'Dump to PKL File')
    pickle.dump(data, open('farsi_handwritten_preprocess.pkl', 'wb', -1))
    pickle.dump(output_dataset['classified'], open('farsi_handwritten_all.pkl', 'wb', -1))


def main():
    file_names = glob.glob(dataset_folder + '*.jpg')
    counter = 0
    id_counter = 0
    all_files_count = len(file_names)
    bad_files = 0
    delete_these = []
    for filename in file_names:
        counter += 1
        img = cv2.imread(filename, 0)
        print()
        print(GREEN, BOLD, '----------------------------------------------------')
        percent = int((counter * 100) / all_files_count)
        print(BOLD, '{}% {}/{} - {}'.format(percent, counter, all_files_count, filename))

        try:
            file_type = filename.split('/')[-1].split('_')[-1][0]
            s_id = filename.split('/')[-1].split('_')[0]
            file_type = True if file_type == '1' else False
            aruco_positions = findAruco(img)
        except Exception as e:
            print(RED, '✖ ️-> ', 'cant find all aruco in {} DELETE'.format(filename))
            delete_these.append(filename)
            bad_files += 1
            continue

        img = doPerspectiveAndSaveRow(aruco_positions, img, filename)
        print(GREEN, '✔️ -> ', 'Row Image Exported.')

        # founded_corners, nC = findCorners(img)
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        #
        # for i in founded_corners:
        #     cv2.circle(img, (i[0], i[1]), 1, (0, 0, 255), thickness=-1)

        # Extract Tiles

        try:
            x_lines, y_lines = findXAndYLines(img)
        except Exception as e:
            print(RED, '✖ ️-> ', 'cant find all lines in {}'.format(filename))
            delete_these.append(filename)
            bad_files += 1
            continue

        for i in range(len(x_lines) + 1):
            y_start = 0
            y_end = len(y_lines) + 1

            if i <= 1:
                y_start = 2
                y_end = len(y_lines) - 1
            if i >= len(x_lines) - 1:
                y_start = 2
                y_end = len(y_lines) - 1

            for j in range(y_start, y_end):
                start = [0, 0]
                end = [0, 0]

                if i > 0:
                    start[0] = x_lines[i - 1]
                if j > 0:
                    start[1] = y_lines[j - 1]

                if i == len(x_lines):
                    end[0] = img_width
                else:
                    end[0] = x_lines[i]

                if j == len(y_lines):
                    end[1] = img_height
                else:
                    end[1] = y_lines[j]

                id_counter += 1
                try:
                    saveTile(img, file_type, j, id_counter, start, end, s_id)
                    if s_id not in s_id_to_file:
                        s_id_to_file[s_id] = [[id_counter, filename]]
                    else:
                        s_id_to_file[s_id].append([id_counter, filename])
                except Exception as ee:
                    print(RED, '✖ ️-> ', 'Save failed ({},{})'.format(i, j))
                    continue

        print(GREEN, '✔️ -> ', 'Tiles exported')

    print(GREEN, BOLD, '----------------------------------------------------')
    print(GREEN, '🗂 -> ', '{} Files | {} Bad | {} Data' \
          .format(all_files_count, bad_files, id_counter))

    dump_to_pkl()

    import json
    json = json.dumps(s_id_to_file)
    f = open("id_to_file_map.json", "w")
    f.write(json)
    f.close()

    if len(delete_these) > 0:
        print('sudo rm {}'.format(' '.join(delete_these)))


if __name__ == '__main__':

    # Make Output Folder
    out_folder = os.path.join(output_folder)
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    if not os.path.exists(out_folder + 'row'):
        os.mkdir(out_folder + 'row')
    if not os.path.exists(out_folder + 'extracted'):
        os.mkdir(out_folder + 'extracted')
    for i in range(42):
        if not os.path.exists(out_folder + 'extracted/' + str(i)):
            os.mkdir(out_folder + 'extracted/' + str(i))

    main()