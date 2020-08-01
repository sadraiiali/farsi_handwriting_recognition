import glob
import os
import cv2
import numpy as np

output_folder = './out/'
input_folder = './examples/'


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


def saveImage(img, filename, j, m, last_start, last_end):
    prefix = ''
    if j == 0:
        prefix = 'ID'
    elif j == 1:
        prefix = 'FN'
    elif j == 2:
        prefix = 'LN'
    elif j == 3:
        prefix = 'PHD'
    elif j == 4:
        prefix = 'MS'
    elif j == 5:
        prefix = 'BS'
    folder_name = filename.split('/')[-1].split('.')[0]

    if j > 2:
        output_filename = output_folder + folder_name + '/' + prefix + '.png'
    else:
        output_filename = output_folder + folder_name + '/' + prefix + str(m + 1) + '.png'
    print(output_filename)

    folder = os.path.join(output_folder + folder_name)
    if not os.path.exists(folder):
        os.mkdir(folder)

    output_img = img[last_start[1]:last_end[1], last_start[0]:last_end[0]]
    # cv2.imshow('corners', output_img)
    # cv2.waitKey(0)  # press any key
    cv2.imwrite(output_filename, output_img)


def main():
    file_names = glob.glob(input_folder + '*.jpg')
    for filename in file_names:
        img = cv2.imread(filename, 0)
        aruco_positions = findAruco(img)

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
            np.array([148, 312], np.float), # Is MS Corner
            np.array([289, 312], np.float)  # Is BC Corner
        ]
        detected_corners = [None, None, None, None, None, None]
        max_dist = 10

        for i in range(1, nC):
            tmp_corner = founded_corners[i, :]
            cv2.circle(img, (int(tmp_corner[0]), int(tmp_corner[1])), 1, (255, 255, 255))
            for j in range(len(corners)):
                cv2.circle(img, (int(corners[j][0]), int(corners[j][1])), 2, (255, 255, 255))
                if np.linalg.norm(corners[j] - tmp_corner) < max_dist:
                    detected_corners[j] = tmp_corner
                    last_start = int(tmp_corner[0]), int(tmp_corner[1])
                    if j < 3:
                        last_end = int(tmp_corner[0]) + 44, int(tmp_corner[1]) + 44
                        cv2.rectangle(img, last_start, last_end, (255, 255, 255))
                        saveImage(img, filename, j, 0, last_start, last_end)
                        for m in range(1, 8):
                            last_start = last_start[0] + 43, last_start[1]
                            last_end = last_end[0] + 43, last_end[1]
                            cv2.rectangle(img, last_start, last_end, (255, 255, 255))
                            saveImage(img, filename, j, m, last_start, last_end)
                    else:
                        last_end = int(tmp_corner[0]) + 15, int(tmp_corner[1]) + 15
                        cv2.rectangle(img, last_start, last_end, (255, 255, 255))
                        saveImage(img, filename, j, 0, last_start, last_end)

        cv2.imshow('corners', img)
        cv2.imwrite('frame.png', img)
        cv2.waitKey(0)  # press any key


if __name__ == '__main__':

    # Make Output Folder
    out_folder = os.path.join(output_folder)
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    main()
