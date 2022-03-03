import cv2
import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable


INPUT_REFERENCE_IMAGE = './reference.jpg'
INPUT_VIDEO_FILE = './input.mov'


def extract_frames():
    original_frame_matrices = []
    grayscale_frame_matrices = []
    vidcap = cv2.VideoCapture(INPUT_VIDEO_FILE)
    while True:
        success, image = vidcap.read()
        if not success:
            break
        original_frame_matrices.append(image)
        grayscale_frame_matrices.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))  # converted to grayscale
    return vidcap, np.asarray(original_frame_matrices), np.asarray(grayscale_frame_matrices)


def get_reference_image():
    return cv2.cvtColor(cv2.imread(INPUT_REFERENCE_IMAGE), cv2.COLOR_BGR2GRAY)


def create_video(vidcap, frame_matrices, method):
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    height, width, layers = frame_matrices[0].shape
    outputV = cv2.VideoWriter('output_' + method + '.mov', fourcc, fps, (width, height))
    for frame in frame_matrices:
        outputV.write(frame)
    outputV.release()

class System:
    def __init__(self, original_frame_matrices, grayscale_frame_matrices, reference_image_matrix):

        self.original_frame_matrices = original_frame_matrices
        self.grayscale_frame_matrices = grayscale_frame_matrices
        self.reference_image_matrix = reference_image_matrix
        self.reference_image_matrix_height, self.reference_image_matrix_width =  self.reference_image_matrix.shape
        self.frame_matrix_height, self.frame_matrix_width = self.grayscale_frame_matrices[0].shape




    def calculate_cost(self, reference_image_matrix, block_matrix):
        return np.sum((reference_image_matrix / 255.0 - block_matrix / 255.0) ** 2)

    def is_valid_location(self, i, j, reference_image_matrix, frame_matrix):
        reference_image_matrix_height, reference_image_matrix_width = reference_image_matrix.shape
        frame_matrix_height, frame_matrix_width = frame_matrix.shape
        if i >= 0 and i + reference_image_matrix_height < frame_matrix_height and j >= 0 and j + reference_image_matrix_width < frame_matrix_width:
            return True
        return False



    def exhaustive_search_on_submatrix(self, reference_image_matrix, grayscale_frame_matrix, previous_best_location, p):
        previous_best_height, previous_best_width = previous_best_location
        reference_image_matrix_height, reference_image_matrix_width =  reference_image_matrix.shape
        minimum_value = np.inf
        new_best_location = -1, -1

        counter = 0

        for i in range(previous_best_height - p, previous_best_height + p + 1):
            for j in range(previous_best_width - p, previous_best_width + p + 1):
                if not self.is_valid_location(i, j, reference_image_matrix, grayscale_frame_matrix):
                    continue
                counter += 1
                block_matrix = grayscale_frame_matrix[i : i + reference_image_matrix_height, j : j + reference_image_matrix_width]
                value = self.calculate_cost(reference_image_matrix, block_matrix)
                if value < minimum_value:
                    minimum_value = value
                    new_best_location = i , j

        return new_best_location, counter

    def exhaustive_search_on_full_matrix(self, reference_image_matrix, grayscale_frame_matrix):
        reference_image_matrix_height, reference_image_matrix_width =  reference_image_matrix.shape
        frame_matrix_height, frame_matrix_width = grayscale_frame_matrix.shape
        best_location = -1, -1
        minimum_value = np.inf

        counter = 0
        for i in range(frame_matrix_height - reference_image_matrix_height + 1):
            for j in range(frame_matrix_width - reference_image_matrix_width + 1):
                counter += 1
                block_matrix = grayscale_frame_matrix[i : i + reference_image_matrix_height, j : j + reference_image_matrix_width]
                value = self.calculate_cost(reference_image_matrix, block_matrix)
                if value < minimum_value:
                    minimum_value = value
                    best_location = i , j

        return best_location, counter

    def d2_logarithmic_search(self, reference_image_matrix, grayscale_frame_matrix, previous_best_location, p):
        previous_best_height, previous_best_width = previous_best_location
        reference_image_matrix_height, reference_image_matrix_width =  reference_image_matrix.shape
        minimum_value = np.inf
        new_best_location = -1, -1

        k = np.int(np.ceil(np.log2(p)))
        d = 2 ** (k - 1)
        p=p//2

        counter = 0

        while d > 1:
            for i in range(previous_best_height - d, previous_best_height + d + 1, d):
                for j in range(previous_best_width - d, previous_best_width + d + 1, d):
                    if not self.is_valid_location(i, j, reference_image_matrix, grayscale_frame_matrix):
                        continue
                    counter += 1
                    block_matrix = grayscale_frame_matrix[i : i + reference_image_matrix_height, j : j + reference_image_matrix_width]
                    value = self.calculate_cost(reference_image_matrix, block_matrix)
                    if value < minimum_value:
                        minimum_value = value
                        new_best_location = i , j
            k = np.int(np.ceil(np.log2(p)))
            d = 2 ** (k - 1)
            p =p// 2

        return new_best_location, counter




    def search(self,vidcap, method, p):
        frame_matrices = self.original_frame_matrices.copy()
        best_location, search_counter = self.exhaustive_search_on_full_matrix(self.reference_image_matrix, self.grayscale_frame_matrices[0])
        total_search_counter = search_counter
        for i in range(1, len(frame_matrices)):
            if method == "exhaustive":
                best_location_top_left, search_counter = self.exhaustive_search_on_submatrix(self.reference_image_matrix, self.grayscale_frame_matrices[i], best_location, p)
            elif method == "2d_logarithmic":
                best_location_top_left, search_counter = self.d2_logarithmic_search(self.reference_image_matrix, self.grayscale_frame_matrices[i], best_location, p)


            print(best_location_top_left)
            best_location_top_left = best_location_top_left[::-1]
            print("after")
            print(best_location_top_left)
            best_location_bottom_right = best_location_top_left[0] + self.reference_image_matrix_width,best_location_top_left[1] + self.reference_image_matrix_height
            color = (0, 0, 255)
            thickness = 3
            cv2.rectangle(frame_matrices[i], best_location_top_left, best_location_bottom_right, color, thickness)


            best_location = best_location_top_left[::-1]
            total_search_counter += search_counter
        create_video(vidcap, frame_matrices, method)

        return total_search_counter


vidcap, original_frame_matrices, grayscale_frame_matrices = extract_frames()
reference_image_matrix = get_reference_image()
system = System( original_frame_matrices, grayscale_frame_matrices, reference_image_matrix)
number_of_frames = len(original_frame_matrices)
print(number_of_frames)

exhaustive_list = []
logarithmic_list = []


p_start = 5
p_end = 10
for p in range(p_start, p_end + 1):
    exhaustive_search_counter = system.search(vidcap,method = "exhaustive", p = p)
    logarithmic_search_counter = system.search(vidcap,method = "2d_logarithmic", p = p)

    exhaustive_list.append((p, exhaustive_search_counter / number_of_frames))
    logarithmic_list.append((p, logarithmic_search_counter / number_of_frames))

exhaustive_array = np.asarray(exhaustive_list)
logarithmic_array = np.asarray(logarithmic_list)


plt.plot(exhaustive_array[:,0], exhaustive_array[:,1], "-g", label = "Exhaustive")
plt.plot(logarithmic_array[:,0], logarithmic_array[:,1], "-r", label = "2D Log")


plt.xlabel('p', fontsize = 12)
plt.ylabel('estimation of performance', fontsize = 12)
plt.legend(loc = "upper left")

plt.show()

pretty_table = PrettyTable(['p', 'Exhaustive', '2D Log'])
for i in range(len(exhaustive_array)):
    pretty_table.add_row([exhaustive_array[i][0], exhaustive_array[i][1] , logarithmic_array[i][1]])

with open('1605117.txt', 'w') as f:
    f.write(pretty_table.get_string())

