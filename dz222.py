import numpy as np
import random
import math
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
image1= cv2.imread(r"C:\Users\yaneg\.git\py\dz2\iphone5.jfif")
image2= cv2.imread(r"C:\Users\yaneg\.git\py\dz2\iphone6.jfif")


#преобразуем изображение в оттенки серого
def rgb_to_grayscale(image):
    height = len(image)
    width = len(image[0])
    gray_image = [[0] * width for _ in range(height)]
    
    for i in range(height):
        for j in range(width):
            r, g, b = image[i][j]
            gray_image[i][j] = int(0.2989 * r + 0.5870 * g + 0.1140 * b)  # Приведение к градациям серого
            
    return gray_image
image1=rgb_to_grayscale(image1)
image2=rgb_to_grayscale(image2)

#реализуем улучшение устойчивости к шуму и искажениям
def normalize_brightness(image):
    # Вычисление среднего значения яркости
    total_brightness = 0
    for row in image:
        for pixel in row:
            total_brightness += pixel
    mean_brightness = total_brightness / (len(image) * len(image[0]))

    # Нормализация яркости
    normalized_image = [[int(pixel - mean_brightness + 128) for pixel in row] for row in image]
    return normalized_image


#создадим ядро гауса
def gaussian_kernel(size, sigma):
    kernel = [[0 for _ in range(size)] for _ in range(size)]
    center = size // 2
    for y in range(size):
        for x in range(size):
            distance_squared = (x - center) ** 2 + (y - center) ** 2
            kernel[y][x] = math.exp(-distance_squared / (2 * sigma ** 2))
    normalization_factor = sum(sum(row) for row in kernel)
    for y in range(size):
        for x in range(size):
            kernel[y][x] /= normalization_factor
    return kernel






# # осуществим мединную фильтрацию изображения
# def median_filter(image, window_size):
#     image_height, image_width = len(image), len(image[0])
#     filtered_image = [[0 for _ in range(image_width)] for _ in range(image_height)]
#     for y in range(window_size // 2, image_height - window_size // 2):
#         for x in range(window_size // 2, image_width - window_size // 2):
#             window = [image[y + dy][x + dx] for dy in range(-window_size // 2, window_size // 2 + 1) for dx in range(-window_size // 2, window_size // 2 + 1)]
#             window.sort()
#             filtered_image[y][x] = window[len(window) // 2]
#     return filtered_image

normalized_image1 = normalize_brightness(image1)
normalized_image2 = normalize_brightness(image2)
    # Размытие Гауссовым фильтром
# для начала напишем функцию для свертки изображения с ядром
def convolve2d(image, kernel):
    kernel_height, kernel_width = len(kernel), len(kernel[0])
    image_height, image_width = len(image), len(image[0])
    output = [[0 for _ in range(image_width - kernel_width + 1)] for _ in range(image_height - kernel_height + 1)]
    #применение свертки
    for y in range(len(output)):
        for x in range(len(output[0])):
            for ky in range(kernel_height):
                for kx in range(kernel_width):
                    output[y][x] += image[y + ky][x + kx] * kernel[ky][kx]
    return output
#применяем фильтр
def gaussian_blur(image, kernel_size, sigma):
    kernel = gaussian_kernel(kernel_size, sigma)
    return convolve2d(image, kernel)


#применяем гаусовское размытие
blurred_image1 = gaussian_blur(normalized_image1, kernel_size=5, sigma=1)
blurred_image2 = gaussian_blur(normalized_image2, kernel_size=5, sigma=1)

#вывод изображения
image11 = np.array(blurred_image1, dtype=np.uint8)
cv2.imshow("blurred_image1", image11)
cv2.waitKey(0)
cv2.destroyAllWindows()



#Перейдем к обнаружению ключевых точек, мной будет использоваться алгоритм Хариса для обнаружения углов

# #еще раз напишем функцию свертки для дальнейшего подсчета градиентов
def convolve(image, kernel):
    kernel_height = len(kernel)
    kernel_width = len(kernel[0])
    img_height = len(image)
    img_width = len(image[0])
    
    # Размеры выходного изображения
    output = [[0 for _ in range(img_width)] for _ in range(img_height)]
    
    # Применение свертки
    for y in range(img_height):
        for x in range(img_width):
            conv_sum = 0
            for ky in range(-kernel_height // 2, kernel_height // 2 + 1):
                for kx in range(-kernel_width // 2, kernel_width // 2 + 1):
                    ixy = y + ky
                    jxy = x + kx
                    if 0 <= ixy < img_height and 0 <= jxy < img_width:
                        conv_sum += image[ixy][jxy] * kernel[ky + kernel_height // 2][kx + kernel_width // 2]
            output[y][x] = conv_sum
    return output

# посчитаем градиенты
def compute_gradients(image):
    
    sobel_x = [[-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]]
    
    sobel_y = [[1, 2, 1],
                [0, 0, 0],
                [-1, -2, -1]]
    
    Ix = convolve(image, sobel_x)
    Iy = convolve(image, sobel_y)

    return Ix, Iy

#вычислим ответ хариса для обнаружения углов
def compute_harris_response(Ix, Iy, k=0.04):
    height = len(Ix)
    width = len(Ix[0])
    R = [[0 for _ in range(width)] for _ in range(height)]
    
    #просуммируем градиенты и вычислим матрицу
    for y in range(height):
        for x in range(width):
            A = B = C = D = 0
            
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    ix = x + dx
                    iy = y + dy
                    if 0 <= ix < width and 0 <= iy < height:
                        Ix_val = Ix[iy][ix]
                        Iy_val = Iy[iy][ix]
                        
                        A += Ix_val * Ix_val
                        B += Ix_val * Iy_val
                        C += Iy_val * Iy_val
            
            det = A * C - B * B
            trace = A + C
            R[y][x] = det - k * (trace ** 2)
    
    return R


#Теперь необходим реализовать функцию подавления нерелевантных углов в массиве значений R. Нам необходимо записать только локальные максимумы

def non_max_suppression(R, threshold):
    height = len(R)
    width = len(R[0])
    corners = [[0 for _ in range(width)] for _ in range(height)]
    corner_coordinates = []
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if R[y][x] > threshold:
                is_max = True
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        if not (dy == 0 and dx == 0):
                            if R[y][x] < R[y + dy][x + dx]:
                                is_max = False
                if is_max:
                    corners[y][x] = 1  # Отметить как угол
                    corner_coordinates.append((x, y)) # надо найти координату
                    
    return corner_coordinates
                
#изображение без обработки
Ix1, Iy1 = compute_gradients(image1)
R1 = compute_harris_response(Ix1, Iy1, k=0.04)
corners1 = non_max_suppression(R1, threshold=100)
Ix2, Iy2 = compute_gradients(image2)
R2 = compute_harris_response(Ix2, Iy2, k=0.04)
corners2 = non_max_suppression(R2, threshold=100)
#изображение с обработкой
Ix11, Iy11 = compute_gradients(blurred_image1)
R11 = compute_harris_response(Ix11, Iy11, k=0.04)
corners11 = non_max_suppression(R11, threshold=100)
Ix22, Iy22 = compute_gradients(blurred_image2)
R22 = compute_harris_response(Ix2, Iy2, k=0.04)
corners22 = non_max_suppression(R22, threshold=100)


#Вывод результатов
# for row in corners:
#     print(row)
    
#найдем описание   
def extract_descriptor(image, keypoints, patch_size=5):
    descriptors = []
    offset = patch_size // 2
    
    for kp in keypoints:
        x, y = int(kp[0]), int(kp[1])
        
        # Убедиться, что не выходим за границы изображения
        if (x - offset < 0 or x + offset >= len(image[0]) or
                y - offset < 0 or y + offset >= len(image)):
            continue
        
        # Извлечение области
        patch = []
        for i in range(y - offset, y + offset + 1):
            patch_row = image[i][x - offset:x + offset + 1]
            patch.append(patch_row)
        
        # Превращаем 2D-область в 1D-вектор
        descriptor = [pixel for row in patch for pixel in row]
        
        # Нормализуем 
        norm = sum(value * 2 for value in descriptor) * 0.5
        if norm > 0:  # Избежать деления на ноль
            descriptor = [value / norm for value in descriptor]
        
        descriptors.append(descriptor)
    
    return descriptors

descriptor1=extract_descriptor(image1, corners1, patch_size=5)
descriptor2=extract_descriptor(image2, corners2, patch_size=5)

descriptor11=extract_descriptor(blurred_image1, corners11, patch_size=5)
descriptor22=extract_descriptor(blurred_image2, corners22, patch_size=5)


#сопоставим ключевые точки между изображениями



def euclidean_distance(desc1, desc2):
    #Вычисляет евклидово расстояние между двумя дескрипторами.
    sum_squared_diff = 0
    for i in range(len(desc1)):
        sum_squared_diff += (desc1[i] - desc2[i]) ** 2
    return sum_squared_diff ** 0.5

def match_keypoints(descriptors1, descriptors2):
    #Сопоставляет ключевые точки между двумя изображениями.
    matches = []
    for i, desc1 in enumerate(descriptors1):
        best_match_index = -1
        best_match_distance = float('inf')
        for j, desc2 in enumerate(descriptors2):
            distance = euclidean_distance(desc1, desc2)
            if distance < best_match_distance:
                best_match_index = j
                best_match_distance = distance
        matches.append((i, best_match_index))
    return matches


# реализуем вариант алгоритма RANSAC
def ransac_filter(matches, keypoints1, keypoints2, threshold=5):
    best_model = None
    best_inliers_count = 0
    for _ in range(100): # Количество итераций RANSAC
        # Выбираем случайную выборку из 4 соответствий
        random_indices = [i for i in range(len(matches))]
        random_indices = random_indices[:4]
        random_matches = [matches[i] for i in random_indices]

        # Вычисляем модель преобразования 
        # на основе случайной выборки
        model = compute_transform(keypoints1, keypoints2, random_matches)

        # Проверяем, сколько соответствий удовлетворяют модели
        inliers_count = 0
        for i, match in enumerate(matches):
            x1, y1 = keypoints1[match[0]][:2]
            x2, y2 = keypoints2[match[1]][:2]
            transformed_point = apply_transform(model, x2, y2)
            distance = ((x1 - transformed_point[0])**2 + (y1 - transformed_point[1])**2) ** 0.5
            if distance < threshold:
                inliers_count += 1

        # Сохраняем модель, если она лучше всех предыдущих
        if inliers_count > best_inliers_count:
            best_model = model
            best_inliers_count = inliers_count

    # Возвращаем список индексов инлайеров
    inliers = []
    for i, match in enumerate(matches):
        x1, y1 = keypoints1[match[0]][:2]
        x2, y2 = keypoints2[match[1]][:2]
        transformed_point = apply_transform(best_model, x2, y2)
        distance = ((x1 - transformed_point[0])**2 + (y1 - transformed_point[1])**2) ** 0.5
        if distance < threshold:
            inliers.append(i)
    return inliers
# функция для транспонированной матрицы
def transpose(matrix):
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]


# функция для умножения матрицы
def matrix_multiply(A, B):
    result = [[0] * len(B[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    return result

def vector_multiply(matrix, vector):
    # векторное умножение
    return [sum(matrix[i][j] * vector[j] for j in range(len(vector))) for i in range(len(matrix))]

#решение системы уравнений методом наименьших квадратов
def least_squares(A, b):
    At = transpose(A)
    AtA = matrix_multiply(At, A)
    Atb = vector_multiply(At, b)
    n = len(AtA)
    x = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            factor = AtA[j][i] / AtA[i][i]
            for k in range(i, n):
                AtA[j][k] -= factor * AtA[i][k]
            Atb[j] -= factor * Atb[i]
    for i in range(n - 1, -1, -1):
        x[i] = Atb[i]
        for j in range(i + 1, n):
            x[i] -= AtA[i][j] * x[j]
        x[i] /= AtA[i][i]
    
    return x

def compute_transform(keypoints1, keypoints2, matches):
    # Создание матрицы A (3x6) для аффинного преобразования
    A = []
    for i in range(len(matches)):
        x1, y1 = keypoints1[matches[i][0]][:2]
        x2, y2 = keypoints2[matches[i][1]][:2]
        A.append([x2, y2, 1, 0, 0, 0])
        A.append([0, 0, 0, x2, y2, 1])

    # Создание вектора b (6x1) с координатами ключевых точек из первого изображения
    b = []
    for i in range(len(matches)):
        x1, y1 = keypoints1[matches[i][0]][:2]
        b.append(x1)
        b.append(y1)

    # Решение системы уравнений Ax = b методом наименьших квадратов
    x = least_squares(A, b)
    transform_matrix = [[x[0], x[1], x[2]],
                        [x[3], x[4], x[5]],
                        [0, 0, 1]]
    return transform_matrix

# применим афинное преобразование
def apply_transform(transform_matrix, x, y):
    # Умножение точки на матрицу преобразования

    x_transformed = transform_matrix[0][0] * x + transform_matrix[0][1] * y + transform_matrix[0][2]
    y_transformed = transform_matrix[1][0] * x + transform_matrix[1][1] * y + transform_matrix[1][2]
    return x_transformed, y_transformed
matches=match_keypoints(descriptor1, descriptor2)
inliers=ransac_filter(matches, descriptor1, descriptor2)
#обработанное изображение
matches1=match_keypoints(descriptor11, descriptor22)
inliers1=ransac_filter(matches1, descriptor11, descriptor22)


print(f"Найдено {len(inliers)} соответствий.")
for i in inliers:
    print(f"Соответствие {i}: {matches[i]}")


print(f"Найдено {len(inliers1)} соответствий.")
for i in inliers1:
    print(f"Соответствие {i}: {matches1[i]}")


# # в качестве результата предварительной обработки изображений, можем увидеть существенное снижение количество соответствий,
# # что говорит о том, что количество ошибок, связанных с неправильной идентификацией соответсвий могло снизиться


