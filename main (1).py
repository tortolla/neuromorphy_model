import argparse
import numpy as np
from impulse import *
from object import *
import imageio
from scipy import ndimage
import numpy as np
from matplotlib.image import imread
from scipy import ndimage
import matplotlib.pyplot as plt
import imageio
from PIL import Image
import PIL
import os



# df = pd.DataFrame()
#
# "создаем картинку для маленького квадратика"
#
# pic = np.zeros((29,29))
# pic_gauss = np.zeros((29,29))
#
# "Создаем картинки"
#
# test = Figure(pic_gauss, 29, 29)
# idol = Figure(pic, 29, 29)
#
# "Заполняем Гаусс шумом"
#
# test.gauss_start(110, 20)
# pic_gauss = test.add_round_shape_gauss(7, 10, 10, 235, 10)
# pic = idol.add_round_shape_gauss(7, 10, 10, 255, 0)
#
#
# test.view()
# test.save('input','/Users/tortolla/Desktop/start')
# idol.save('idol', '/Users/tortolla/Desktop/start')
#
#
# structures_10 = {}
# structures_40 = {}
# structures_100 = {}
# structures_1000 = {}
# structures_2000 = {}
#
#
#
# number  = 0
# pic_in_time = np.zeros((29,29))
#
#
# "Нужны другие коэффициенты для спайкового поезда"
#
#

#
#
# "Создаем обьект для каждой структуры"
#
#
# #np.savetxt(f'/Users/tortolla/Desktop/for_victor/start.txt', test.pic)
# #print(test.pic)


# for i in range(0, 29):
#     for j in range(0, 29):
#
#         struct_10 = Structure.__new__(Structure)
#         struct_10.__init__(f"{i}_{j}", i, j, test.image, pow2, coef2, 10)
#         structures_10[struct_10.name] = struct_10
#         struct_10.event_array = struct_10.get_self_event_array()
#
#         struct_40 = Structure.__new__(Structure)
#         struct_40.__init__(f"{i}_{j}", i, j, test.image, pow2, coef2, 40)
#         structures_40[struct_40.name] = struct_40
#         struct_40.event_array = struct_40.get_self_event_array()
#
#         struct_100 = Structure.__new__(Structure)
#         struct_100.__init__(f"{i}_{j}", i, j, test.image, pow2, coef2, 100)
#         structures_100[struct_100.name] = struct_100
#         struct_100.event_array = struct_100.get_self_event_array()
#
#         struct_1000 = Structure.__new__(Structure)
#         struct_1000.__init__(f"{i}_{j}", i, j, test.image, pow2, coef2, 1000)
#         structures_1000[struct_1000.name] = struct_1000
#         struct_1000.event_array = struct_1000.get_self_event_array()
#
#         struct_2000 = Structure.__new__(Structure)
#         struct_2000.__init__(f"{i}_{j}", i, j, test.image, pow2, coef2, 2000)
#         structures_2000[struct_2000.name] = struct_2000
#         struct_2000.event_array = struct_2000.get_self_event_array()
#
#
#         #array = struct.get_self_event_array()
#         #df[struct.name] = array.tolist()
#
#
# #df.to_csv('//Users/tortolla/Downloads/9.png')
#
#
#
# "Результат для видео"
# path_10 = r'/Users/tortolla/Desktop/image_10/'
# path_40 = r'/Users/tortolla/Desktop/image_40/'
# path_100 = r'/Users/tortolla/Desktop/image_100/'
# path_1000 = r'/Users/tortolla/Desktop/image_1000/'
# path_2000 = r'/Users/tortolla/Desktop/image_2000/'
#
# over_look(structures_10, 1, 'experiment_10', 29, 29, path_10)
# over_look(structures_40, 10, 'experiment_40', 29, 29, path_40)
# over_look(structures_100, 20, 'experiment_100', 29, 29, path_100)
# over_look(structures_1000, 200, 'experiment_1000', 29, 29, path_1000)
# over_look(structures_2000, 400, 'experiment_2000', 29, 29, path_2000)
#
#
# for i in range(0, 29):
#     for j in range(0, 29):
#         count = structures_10[f'{i}_{j}'].sgo_event()
#         print(f"count_10:")
#
# for i in range(0, 29):
#     for j in range(0, 29):
#         count = structures_40[f'{i}_{j}'].sgo_event()
#         print(f"count_40:")
#
# for i in range(0, 29):
#     for j in range(0, 29):
#         count = structures_100[f'{i}_{j}'].sgo_event()
#         print(f"count_100:")
#
# for i in range(0, 29):
#     for j in range(0, 29):
#         count = structures_1000[f'{i}_{j}'].sgo_event()
#         print(f"count_1000:")
#
# for i in range(0, 29):
#     for j in range(0, 29):
#         count = structures_2000[f'{i}_{j}'].sgo_event()
#         print(f"count_2000:")

pow2 = 8
coef2 = 1.7
numo = 0
# создаем пустой словарь
files_dict = []


# получаем список файлов из папки
files = os.listdir('/Users/tortolla/Desktop/geo_input')
print(files)

#проходим по списку файлов
for file in files:
    path = os.path.join('/Users/tortolla/Desktop/geo_input', file)
    files_dict.append(path)


for big_path in files_dict:

# "Переводим в серый цвет"
    pic_big, pic_big_1 = convert_to_grayscale(big_path)
    pic_big = np.array((pic_big))
    pic_big = pic_big.transpose()
    print(pic_big)
    a, b = pic_big.shape
    print(pic_big.shape)
#
    gray = Figure(pic_big, a, b)
    image = Image.open(big_path)
    stupid = PIL.Image.open(big_path)
    s = stupid.convert('L')
    gray.save('geo_gray_start', '/Users/tortolla/Desktop/start_1')
#
# "Создаем список для обьектов класса структур"
    structures_1 = {}
    number  = 0
    pic_in_time = np.zeros((a,b))
#
#
# "Коэффициенты для спайковой цепочки"
#
    pow1 = 8
    coef1 = 1.7

#
# "Создаем обьекты класса и записываем в список"
#
    #big_structures_10 = {}
    #big_structures_40 = {}
    big_structures_100 = {}
    big_structures_1000 = {}
    #big_structures_2000 = {}

    for i in range(0, a):
        for j in range(0, b):

            # big_struct_10 = Structure.__new__(Structure)
            # big_struct_10.__init__(f"{i}_{j}", i, j, gray.image, pow2, coef2, 10)
            # big_structures_10[big_struct_10.name] = big_struct_10
            # big_struct_10.event_array = big_struct_10.get_self_event_array()
            #
            # big_struct_40 = Structure.__new__(Structure)
            # big_struct_40.__init__(f"{i}_{j}", i, j, gray.image, pow2, coef2, 40)
            # big_structures_40[big_struct_40.name] = big_struct_40
            # big_struct_40.event_array = big_struct_40.get_self_event_array()

            big_struct_100 = Structure.__new__(Structure)
            big_struct_100.__init__(f"{i}_{j}", i, j, gray.image, pow2, coef2, 100)
            big_structures_100[big_struct_100.name] = big_struct_100
            big_struct_100.event_array = big_struct_100.get_self_event_array()

            big_struct_1000 = Structure.__new__(Structure)
            big_struct_1000.__init__(f"{i}_{j}", i, j, gray.image, pow2, coef2, 1000)
            big_structures_1000[big_struct_1000.name] = big_struct_1000
            big_struct_1000.event_array = big_struct_1000.get_self_event_array()

            print(f"{i}_{j}")

            # big_struct_2000 = Structure.__new__(Structure)
            # big_struct_2000.__init__(f"{i}_{j}", i, j, gray.image, pow2, coef2, 2000)
            # big_structures_2000[big_struct_2000.name] = big_struct_2000
            # big_struct_2000.event_array = big_struct_2000.get_self_event_array()

    #
# "Считаем количество световых импульсов"

    #path_10 = r'/Users/tortolla/Desktop/image_10_1/'
    #path_40 = r'/Users/tortolla/Desktop/image_40_1/'
    path_100 = r'/Users/tortolla/Desktop/geo/'
    path_1000 = r'/Users/tortolla/Desktop/geo/'
    #path_2000 = r'/Users/tortolla/Desktop/image_2000_1/'
#

#
# "Смотрим картинку через 55000 секунд"
    name  = 'p' + str(numo)

    #over_look(big_structures_10, 1, name, a, b, path_10)
    #over_look(big_structures_40, 10, name, a, b, path_40)
    over_look(big_structures_100, 20, name, a, b, path_100)
    over_look(big_structures_1000, 200, name, a, b, path_1000)
    #over_look(big_structures_2000, 400, name, a, b, path_2000)


    numo += 1
















