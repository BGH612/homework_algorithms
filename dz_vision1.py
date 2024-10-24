import numpy as np
import unittest
import matplotlib.pyplot as plt
from skimage import io, color
import cv2
from PIL import Image

#преобразование в бинарное
def rgb_to_binary(image):
    heigh=len(image)
    width=len(image[0])
    binary=[[0]*width for a in range(heigh)]
    for i in range(heigh):
        for j in range(width):
            r,g,b=image[i][j]
            # воспользуемся функцией для gray_scale
            brightness= r*0.299+g*0.587+b*0.114
            # если результат больше 128, присваиваем 1, в другом случае - 0
            if brightness > 128:
                binary[i][j]=1 
            else:
                binary[i][j]=0
    return binary
# преобразование в gray_scale    


def rgb_to_grayscale(image):
    heigh=len(image)
    width=len(image[0])
    gray_scale=[[0]*width for a in range(heigh)]
    for i in range(heigh):
        for j in range(width):
            r,g,b=image[i][j]
            gray= int(r*0.299+g*0.587+b*0.114)
            gray_scale[i][j]=gray
    return gray_scale
        
    
    
color_image = [
    [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
    [[0, 0, 0], [255, 255, 255], [128, 128, 128]],
    [[255, 255, 0], [0, 255, 255], [255, 0, 255]]
]

#теперь проверим как работает функция

# Преобразование в бинарное изображение
binary_image = rgb_to_binary(color_image)
print("Binary")
for a in binary_image:
    print(a)
#преобразование в gray_scale   
gray_scale= rgb_to_grayscale(color_image)
print('grayscale')
for a in gray_scale:
    print(a)
    

# Реализация алгоритма ОЦУ


# Для начала необходимо преобразовать изображение в гистограмму
def get_histogram(image):
    histogram = [0] * 256
    for pixel in image.getdata():
        histogram[pixel] += 1
    return histogram
# получим оптимальный порог разделения ОЦУ
def otsu_threshold(image):
    histogram = get_histogram(image)
    total_pixels = sum(histogram)
    
    # Нормализация гистограммы
    sum_total = sum(i * histogram[i] for i in range(256))
    sum_back = 0
    weight_back = 0
    maximum = 0
    threshold = 0

    for i in range(256):
        # вычислим задний фон
        weight_back += histogram[i]
        if weight_back == 0:
            continue
        # вычисляем объекты  
        weight_fore = total_pixels - weight_back
        if weight_fore == 0:
            break

        sum_back += i * histogram[i]
        sum_fore = sum_total - sum_back

        # Вычисление средних значений и дисперсии
        mean_back = sum_back / weight_back
        mean_fore = sum_fore / weight_fore

        # Вычисление дисперсии
        between_class_variance = weight_back * weight_fore * (mean_back - mean_fore) ** 2

        # Проверка на максимум
        if between_class_variance > maximum:
            maximum = between_class_variance
            threshold = i     
    return threshold

# получим бинаризированное изображение методом оцу
def image_binarized(image,threshold):
    return image.point(lambda p: 255 if p > threshold else 0) 


def main(adress):
    # получение изображения в оттенках серого
    image = Image.open(adress).convert("L")
    
    # получаем порог
    threshold = otsu_threshold(image)
    
    # Бинаризация изображения
    binary_image= image_binarized(image,threshold)
    # вывод изображения
    binary_image.show()
    print(f"Порог Otsu: {threshold}")

#Проведение тестов

# Тест 1, сравнение с библиотекой cv2
#  проверка бинаризации для Дома, изображение включает в себя различные компоненты и цвета
main(r"C:\Users\yaneg\.git\py\house.jpg")
# мы видим как работает бинаризация для дома, дом и его контуры четко отличимы от других фрагментов кадра

# теперь сравним результат со встроенной библиотекой cv2, имеющей модуль, реализующий функцию ОЦУ
image1 = cv2.imread(r"C:\Users\yaneg\.git\py\house.jpg", cv2.IMREAD_GRAYSCALE)

thresh_val, thresh_image = cv2.threshold(image1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(f"Порог Otsu из Cv2:{thresh_val}")

# Для отображения изображения можно использовать следующий код
cv2.imshow('Otsu Thresholding', thresh_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# визуально, можно увидеть, что результат работы встроенной функции аналогичный с моим, более того порог оцу из функции встроенной в библиотеку cv2 равен порогу оцу вычисленному моей функцией


#Тест 2, контрастное изображение
main(r"C:\Users\yaneg\.git\py\contast1.jpg")
#для контрастного изображение в том числе можно увидеть наиболее четкое разделение, несмотря на то, что картина нарисована скорее всего акварелью, однаго в некоторых местах алгоритм немного ошибается, из-за того что краски на картинке получились слишком темные

# Тест 3, возьмем изображение с шумом, темную, ночную улицу
main(r"C:\Users\yaneg\.git\py\noise.jpg")
# на данном изображении мало, что можно определить из объектов, поскольку оно слишком темное, единственное, что видно- окна, и то это результат того, что они излучают свет, таким образом данное изображение разделилось на светлые и темные участки


#Тест 4, простое изображение с шумом
main(r"C:\Users\yaneg\.git\py\noise2.jpg")
# Мы можем увидеть довольно хороший результат, поскольку черты лица и контур лица хорошо видны

# Тест 5, темное изображение
main(r"C:\Users\yaneg\.git\py\darkstreet.jfif")
# на данном изображении немалое количество объектов, однако очертания улицы и объектов довольно хорошо видны

# Тест 6, проверка влияния шума
main(r"C:\Users\yaneg\.git\py\noise3.jpg")
# На данном изображении мы можем увидеть два одинкаовых изображении, при чем одно- с шумом. В целом, общие контуры изображения примерно одинаковые, однако шум все таки немного влияет, заметен на обработанном изображении

#Тест 7, пустое изображение 
main(r"C:\Users\yaneg\.git\py\empty.jpg")
# Результат пустого изображения- другое пустое изображение, с одним цветом, что говорит о правильности работы алгоритма



#тесты 8,9 и 10 с использованием модульного тестирования
# Тест 8, изображение с преобладающим цветом
class TestImageProcessing(unittest.TestCase):

    def setUp(self):
        #создадим изображение, с преобладающим цветом пикселя- 100
        self.image_data = np.zeros((10, 10), dtype=np.uint8)
        self.image_data[0:10, 0:10] = 100  
        self.image_data[5:10, 5:10] = 200  
        
        # переведем формат изображения в пил
        self.test_image = Image.fromarray(self.image_data)
        # проверим правильно ли наш код преобразует изображение в диаграмму, долю распределения для каждого оттенка
    def test_get_histogram(self):
        histogram = get_histogram(self.test_image)
        self.assertEqual(histogram[100], 75)
        self.assertEqual(histogram[200], 25)
        self.assertEqual(histogram[0], 0)
# проверим правильно ли прохоит порог разделения, оптимальный порог здесь -100
    def test_otsu_threshold(self):
        threshold = otsu_threshold(self.test_image)
        self.assertEqual(threshold, 100)
# проверим, что пиксели получают значения 0 или 255
    def test_image_binarized(self):
        threshold = otsu_threshold(self.test_image)
        binarized_image = image_binarized(self.test_image, threshold)
        binarized_data = binarized_image.getdata()
        for pixel in binarized_data:
            self.assertIn(pixel, [0, 255])

# Тест 9, изображение заполненное только одним цветом
class TestImageProcessing1(unittest.TestCase):
    def setUp(self):
        #создадим изображение, заполненное одинаковыми пискелями
        self.image_data = np.zeros((10, 10), dtype=np.uint8) 
        
        # переведем формат изображения в пил
        self.test_image = Image.fromarray(self.image_data)
        # проверим правильно ли наш код преобразует изображение в диаграмму, долю распределения для каждого оттенка
    def test_get_histogram1(self):
        histogram = get_histogram(self.test_image)
        self.assertEqual(histogram[0], 100)
# проверим правильно ли прохоит порог разделения, оптимальный порог здесь -0
    def test_otsu_threshold1(self):
        threshold = otsu_threshold(self.test_image)
        self.assertEqual(threshold, 0)
# проверим, что пиксели получают значения 0 или 255
    def test_image_binarized1(self):
        threshold = otsu_threshold(self.test_image)
        binarized_image = image_binarized(self.test_image, threshold)
        binarized_data = binarized_image.getdata()
        for pixel in binarized_data:
            self.assertIn(pixel, [0, 255])

# тест 10, изображение у которого одинаковое количество и того и другого цвета
class TestImageProcessing2(unittest.TestCase):
    def setUp(self):
        #создадим изображение, заполненное с одинаковым количество пискелей со значением 5 и 10
        self.image_data = np.zeros((10, 10), dtype=np.uint8) 
        self.image_data[0:5, :] = 5  
        self.image_data[5:10, :] = 10
        # переведем формат изображения в пил
        self.test_image = Image.fromarray(self.image_data)
        # проверим правильно ли наш код преобразует изображение в диаграмму, долю распределения для каждого оттенка
    def test_get_histogram1(self):
        histogram = get_histogram(self.test_image)
        self.assertEqual(histogram[5], 50)
        self.assertEqual(histogram[10], 50)
# проверим правильно ли прохоит порог разделения, оптимальный порог здесь -5, который равен первой половине гистограммы
    def test_otsu_threshold1(self):
        threshold = otsu_threshold(self.test_image)
        self.assertEqual(threshold, 5)
# проверим, что пиксели получают значения 0 или 255
    def test_image_binarized1(self):
        threshold = otsu_threshold(self.test_image)
        binarized_image = image_binarized(self.test_image, threshold)
        binarized_data = binarized_image.getdata()
        for pixel in binarized_data:
            self.assertIn(pixel, [0, 255])
unittest.main()   

