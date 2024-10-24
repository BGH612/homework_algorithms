Для запуска данного кода достаточно установить необходимые библиотеки на ваш компьюетр и запустить его, в функции main, необходимо указывать ваш путь до изображения. 


В моей работе сначала задается функция для преобразования изображения rgb в бинарное (rgb_to_binary), для начала код принимает на вход трехмерный массив данных, в котором содержатся данные о значениях red, green и blue для каждого пикселя.
Для вычисления бинарного значения для каждого пикселя мы применяем формулу red*0.299+green*0.587+blue*0.114, результат которой мы сравниваем с числом 128, если значение >128, бинарному значению присваивается 1, в обратном случае 0

После чего я вывожу функцию, для преобразования изображения в gray_scale (rgb_to_grayscale), для начала код принимает на вход трехмерный массив данных, в котором содержатся данные о значениях red, green и blue для каждого пикселя.
Для вычисления значения gray_Scale для каждого пикселя мы применяем формулу red*0.299+green*0.587+blue*0.114, полученное значение является значением gray_scale.

Далее я проверяю, как работают оба алгоритма на тестовом массиве.


После чего я перехожу к реализации алгоритма ОЦУ. Для начала необходимо преобразовать изображение в гистограмму (get_histogram). Здесь создается список histogram, содержащий 256 нулей. 
Для этого я создаю список, каждый индекс в этом списке будет соответствовать значению яркости (градации серого) каждого пикселя в изображении, начиная с 0 и заканчивая 255.
Метод getdata() возвращает последовательность (обычно в виде плоского списка или генератора) всех пикселей изображения. Цикл for проходит по каждому пикселю.
Для каждого пикселя значение pixel используется как индекс в списке histogram, и соответствующее значение увеличивается на 1. Это действие подсчитывает, сколько раз каждое значение яркости встречается в изображении.

Функция otsu_threshold принимает на вход изображение, для начала преобразует его в гистограмму при помощи функции get_histogram, после чего определенить общее количество пикселей total_pixels,
Рассчитать сумму интенсивностей sum_total,
Обозначить переменные для накопления значений sum_back, weight_back, maximum, threshold
При помощи цикла, Увеличиваем вес фонового класса, добавляя количество пикселей с текущей интенсивностью weight_back
Если вес фонового класса равен нулю, продолжаем.
Рассчитываем вес класса объектов weight_fore, как все пиксели минус фон.
Если вес класса объектов равен нулю (всё изображение стало фоном), выходим из цикла, так как больше нет пикселей для анализа.

Обновляем сумму интенсивностей фонового класса sum_back
Вычисляем сумму интенсивностей класс объектов sum_fore
Вычисляем средние значения mean_back и mean_fore, и наконец вычисляем междуклассовой дисперсии
Обновляем максимальную дисперсию Если текущая междуклассовая дисперсия больше between_class_variance, чем ранее найденная maximum, обновляем максимальную дисперсию и устанавливаем текущий порог.
И наконец находим значение оптимального порока разделения threshold для макисмальной дисперсии, после чего применяем его, для бинаризации изображения image_binarized

Теперь напишем функцию main для получения оттенков серого по адресу папки в которой находится изображение.
Она принимает на вход месторасположение изображения в вашем пк и преобразует его в оттенки серого, после чего получает порог ОЦУ, по которому бинаризует изображение и выводит его.

Проведение тестов 

Тест 1 
Сравнение работы моего алгоритма с библиотекой cv2. Проверка бинаризации для Дома, house.jpg, изображение включает в себя различные компоненты и цвета. 
Мы видим как работает бинаризация для дома, дом и его контуры четко отличимы от других фрагментов кадра
Теперь сравним результат со встроенной библиотекой cv2, имеющей модуль, реализующий функцию ОЦУ cv2.threshold

Визуально, можно увидеть, что результат работы встроенной функции аналогичный с моим, более того порог оцу из функции встроенной в библиотеку cv2 равен порогу оцу вычисленному моей функцией

Тест 2
Контрастное изображение contast1.jpg

Для контрастного изображения в том числе можно увидеть наиболее четкое разделение, несмотря на то, что картина нарисована скорее всего акварелью
Однако в некоторых местах алгоритм немного ошибается, из-за того что краски на картинке получились слишком темные

Тест 3
Возьмем изображение с шумом, темную, ночную улицу, noise.jpg
на данном изображении мало, что можно определить из объектов, поскольку оно слишком темное, единственное, что видно- окна, 
И то это результат того, что они излучают свет, таким образом данное изображение разделилось на светлые и темные участки

#Тест 4 
Простое изображение с шумом, noise2.jpg
Мы можем увидеть довольно хороший результат, поскольку черты лица и контур лица хорошо видны

Тест 5
темное изображение, darkstreet.jfif
На данном изображении немалое количество объектов, однако очертания улицы и объектов довольно хорошо видны

Тест 6
проверка влияния шума, noise3.jpg

На данном изображении мы можем увидеть два одинкаовых изображении, при чем одно- с шумом. 
В целом, общие контуры изображения примерно одинаковые, однако шум все таки немного влияет, заметен на обработанном изображении.

Тест 7

Пустое изображение, empty.jpg
Результат пустого изображения- другое пустое изображение, с одним цветом, что говорит о правильности работы алгоритма


Тесты 8,9,10 я проводил с использованием модульного тестирования

Тест 8, изображение с преобладающим цветом

создадим изображение, с преобладающим цветом пикселя- 100
переведем формат изображения в пил
проверим правильно ли наш код преобразует изображение в диаграмму, долю распределения для каждого оттенка
проверим правильно ли прохоит порог разделения, оптимальный порог здесь 100, поскольку пробладающий цвет - 100
проверим, что пиксели получают значения 0 или 255 

