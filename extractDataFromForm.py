import glob
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

input_folder = './fromImages/'
output_model = './learned/'

digitModel = None
lettersModel = None
D_I_SIZE = 28
L_I_SIZE = 32

RED = '\033[91m'
GREEN = '\033[92m'
BOLD = '\033[1m'

farsi_digits_fa = [
    '۰', '۱', '۲', '۳', '۴', '۵', '۶', '۷', '۸', '۹',
    'ا', 'ب', 'پ', 'ت', 'ث', 'ج', 'چ', 'ح', 'خ', 'د',
    'ذ', 'ر', 'ز', 'ژ', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ',
    'ع', 'غ', 'ف', 'ق', 'ک', 'گ', 'ل', 'م', 'ن', 'و',
    'ه', 'ی',
]


# noinspection PyUnresolvedReferences
def findAruco(img):
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters_create()
    marker_corners, marker_ids, rejected_candidates = cv2.aruco.detectMarkers(img, dictionary, parameters=parameters)

    marker_ids = marker_ids.tolist()
    a_33_i = marker_ids.index([33])
    a_34_i = marker_ids.index([34])
    a_35_i = marker_ids.index([35])
    a_36_i = marker_ids.index([36])

    p1 = marker_corners[a_34_i][0][3][0], marker_corners[a_34_i][0][3][1]
    p2 = marker_corners[a_35_i][0][2][0], marker_corners[a_35_i][0][2][1]
    p3 = marker_corners[a_36_i][0][1][0], marker_corners[a_36_i][0][1][1]
    p4 = marker_corners[a_33_i][0][0][0], marker_corners[a_33_i][0][0][1]

    return np.array([p1, p2, p3, p4], dtype=np.float32)


def correctPerspective(img, aruco_positions, correct_form, output_size):
    H = cv2.getPerspectiveTransform(aruco_positions, correct_form)
    J = cv2.warpPerspective(img, H, output_size)
    return J


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


def digitExtract(img, last_start, last_end):
    org_img = img[last_start[1]:last_end[1], last_start[0]:last_end[0]]
    digit_img = cv2.resize(org_img,
                           (D_I_SIZE, D_I_SIZE))

    digit_img = digit_img / 255
    digit_img = digit_img.reshape(1, D_I_SIZE, D_I_SIZE, 1)

    predictions = digitModel.predict(digit_img)
    label = np.argmax(predictions)
    prob = np.max(predictions)
    return str(label)


def lettersExtract(img, last_start, last_end):
    org_img = img[last_start[1] + 5:last_end[1] - 5, last_start[0] + 5:last_end[0] - 5]

    # print(np.var(org_img))
    letters_img = cv2.resize(org_img,
                             (L_I_SIZE, L_I_SIZE))

    letters_img = letters_img / 255
    letters_img = letters_img.reshape(1, L_I_SIZE, L_I_SIZE, 1)

    predictions = lettersModel.predict(letters_img)
    label = np.argmax(predictions)
    prob = np.max(predictions)
    # if prob < 0.7:
    #     print('low',prob)
    #     return 0
    # cv2.imwrite('a.png', org_img)
    # print(farsi_digits_fa[label + 10])
    # input('-----')
    return label


def loadModel():
    global digitModel, lettersModel

    # Digit MODEL
    digitModel = tf.keras.models.Sequential()
    digitModel.add(tf.keras.layers.Conv2D(filters=32,
                                          kernel_size=(5, 5),
                                          padding='same',
                                          activation='relu',
                                          input_shape=(D_I_SIZE, D_I_SIZE, 1)
                                          ))
    digitModel.add(tf.keras.layers.MaxPool2D(strides=2))
    digitModel.add(tf.keras.layers.Conv2D(filters=48,
                                          kernel_size=(5, 5),
                                          padding='valid',
                                          activation='relu'
                                          ))
    digitModel.add(tf.keras.layers.MaxPool2D(strides=2))
    digitModel.add(tf.keras.layers.Flatten())
    digitModel.add(tf.keras.layers.Dense(256,
                                         activation='relu'))
    digitModel.add(tf.keras.layers.Dense(84,
                                         activation='relu'))
    digitModel.add(tf.keras.layers.Dense(10,
                                         activation='softmax'))
    digitModel.build()
    digitModel.load_weights(output_model + "learned_digits.h5")
    # ---------------------------------------------------------------
    # Letters MODEL
    lettersModel = tf.keras.models.Sequential()
    lettersModel.add(tf.keras.layers.Conv2D(filters=20,
                                            kernel_size=(5, 5),
                                            padding='same',
                                            activation='relu',
                                            input_shape=(L_I_SIZE, L_I_SIZE, 1)
                                            ))
    lettersModel.add(tf.keras.layers.MaxPool2D(strides=2))
    lettersModel.add(tf.keras.layers.Conv2D(filters=50,
                                            kernel_size=(5, 5),
                                            padding='valid',
                                            activation='relu'
                                            ))
    lettersModel.add(tf.keras.layers.MaxPool2D(strides=2))
    lettersModel.add(tf.keras.layers.Flatten())

    lettersModel.add(tf.keras.layers.Dense(500,
                                           activation='relu'))
    lettersModel.add(tf.keras.layers.Dense(256,
                                           activation='relu'))
    lettersModel.add(tf.keras.layers.Dense(32,
                                           activation='softmax'))
    lettersModel.build()
    lettersModel.load_weights(output_model + "learned_letters.h5")

    print(GREEN, '-----------------------')
    print(GREEN, '✔️ -> ', 'Models Loaded')
    print(GREEN, '-----------------------')


def main():
    start_time = time.time()
    loadModel()

    print("--- Model Loading time %s ---" % (time.time() - start_time))

    file_names = glob.glob(input_folder + '*.jpg')
    for filename in file_names:
        start_time = time.time()
        img = cv2.imread(filename, 0)
        plt.imshow(img)
        plt.show()
        aruco_positions = findAruco(img)
        s_id = ''
        out_f_name = ''
        out_l_name = ''

        x = 40
        n = int(12.8 * x)
        m = int(12.4 * x)
        output_size = (n, m)
        correct_form = np.array([
            (0, 0), (n, 0),
            (n, m), (0, m)
        ], dtype=np.float32)

        img = correctPerspective(img, aruco_positions, correct_form, output_size)

        founded_corners, nC = findCorners(img)

        corners = [
            np.array([25, 128], np.float),  # Student Id Corner
            np.array([25, 189], np.float),  # First Name Corner
            np.array([25, 248], np.float),  # Last Name Corner
            np.array([50, 312], np.float),  # Is PHD Corner
            np.array([148, 312], np.float),  # Is MS Corner
            np.array([289, 312], np.float)  # Is BC Corner
        ]
        detected_corners = [None, None, None, None, None, None]
        education_selection = [None, None, None]
        max_dist = 10

        for i in range(1, nC):
            tmp_corner = founded_corners[i, :]
            # cv2.circle(img, (int(tmp_corner[0]), int(tmp_corner[1])), 1, (255, 255, 255))
            for j in range(len(corners)):
                # cv2.circle(img, (int(corners[j][0]), int(corners[j][1])), 2, (255, 255, 255))
                if np.linalg.norm(corners[j] - tmp_corner) < max_dist:
                    detected_corners[j] = tmp_corner
                    last_start = int(tmp_corner[0]), int(tmp_corner[1])
                    if j < 3:
                        last_end = int(tmp_corner[0]) + 44, int(tmp_corner[1]) + 44
                        # cv2.rectangle(img, last_start, last_end, (255, 255, 255))
                        for m in range(1, 8):
                            last_start = last_start[0] + 43, last_start[1]
                            last_end = last_end[0] + 43, last_end[1]
                            if j == 0:
                                digit = digitExtract(img, last_start, last_end)
                                s_id += str(digit)
                            elif j == 1:
                                letter = lettersExtract(img, last_start, last_end)
                                out_f_name += farsi_digits_fa[letter + 10]
                            elif j == 2:
                                letter = lettersExtract(img, last_start, last_end)
                                # print(str(letter), end=' ')
                                out_l_name += farsi_digits_fa[letter + 10]

                            # cv2.rectangle(img, last_start, lastoutput_model_end, (255, 255, 255))
                    else:
                        last_end = int(tmp_corner[0]) + 15, int(tmp_corner[1]) + 15
                        detected_img = img[last_start[1] + 5:last_end[1] - 5, last_start[0] + 5:last_end[0] - 5]
                        education_selection[j - 3] = detected_img
                        cv2.rectangle(img, last_start, last_end, (255, 255, 255))

        max_i = 0
        max_intensity = 0
        for i in range(len(education_selection)):
            intensity_sum = np.sum(education_selection[i])
            if intensity_sum >= max_intensity:
                max_intensity = intensity_sum
                max_i = i
        print()
        print(GREEN, 'Image :', filename)
        print(RED, 'Student ID :', GREEN, s_id)
        print(RED, 'Student FirstName :', GREEN, out_f_name[::-1])
        print(RED, 'Student LastName :', GREEN, out_l_name[::-1])
        if max_i == 0:
            print(RED, 'Education', GREEN, 'PHD')
        elif max_i == 1:
            print(RED, 'Education', GREEN, 'MASTER')
        elif max_i == 2:
            print(RED, 'Education', GREEN, 'BACHLOR')
        plt.imshow(img)
        plt.show()
        print(BOLD, "- Answering time: %.2f Sec" % (time.time() - start_time))
        print(GREEN, "-----------------------------------------------------")
        input()

        # cv2.imshow('corners', img)
        # cv2.imwrite('frame.png', img)
        # cv2.waitKey(0)  # press any key


if __name__ == '__main__':
    main()
