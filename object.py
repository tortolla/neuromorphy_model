import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from impulse import *
import math
from impulse import *
import csv
from impulse import *
from object import *
import imageio
from scipy import ndimage
import numpy as np
from matplotlib.image import imread
from scipy import ndimage
import matplotlib.pyplot as plt
import imageio
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import imageio.v2 as imageio


path = r'/Users/tortolla/Desktop/image/'
pic = np.zeros((8, 8))


def over_look(structures, delta, name, width, height, path):

    """
    Function saves image every time frame
    params:
    structures - array of class structure objects
    delta - time frame
    name - name of image
    width - width of image
    height - height of image
    """

    pic_in_time = np.zeros((width, height))
    t = 0
    name_time  = 0
    time = structures["1_1"].time0

    while t <= time:
        for i in range(0, width):
            for j in range(0, height):
                pic_in_time[i][j] = structures[f"{i}_{j}"].cond_in_time(t)

        look = Figure(pic_in_time, width, height)
        look.save(f'{name}_{t}', path)
        print(name_time)
        name_time += delta
        t = t + delta


def convert_to_grayscale(image_path):
    """
    Функция преобразовывает изображение в черно-белое
    params:
    image_path - путь до изображения
    returns:
    pixel_array - массив в форме: [i][j] элемент соответствует яркости [i][j] пикселя
    pixel_array_1 - тот же массив, но в одну строчку
    """


    image = Image.open(image_path)


    grayscale_image = image.convert('L')
    width, height = grayscale_image.size
    grayscale_image.show()


    pixel_values = grayscale_image.getdata()
    pixel_array_1 = np.array(pixel_values)


    pixel_array = np.array(pixel_array_1).reshape((height, width))


    return pixel_array, pixel_array_1




"Class of figures - creating of image, adding noise, saving, screen of output"
class Figure(object):

    """
    The input is image width, image length
    """


    def __init__(self, pic, width, height):

        self.pic = pic
        self.width = width
        self.height = height
        self.image = Image.new("L", (self.width, self.height))
        self.draw = ImageDraw.Draw(self.image)


    def add_random_shape(self, length, max, min):

        """
        Add a random shape to the images - the brightness of each pixel is determined in the range max-min
        using randint()
        param:
        length - side length
        max - maximum brightness of pixel
        min - minimum brightness of pixel
        returns:
        pic - array with pixel brightness of image with random figure on it
        ***
        Also this function changes self.pic, so you probably don't change
        """

        image_shape = pic.shape

        max_x = image_shape[0] - length
        max_y = image_shape[1] - length

        # Генерируем случайные координаты верхнего левого угла фигуры
        start_x = np.random.randint(0, max_x)
        start_y = np.random.randint(0, max_y)

        # Задаем случайную фигуру на изображении
        for i in range(start_x, start_x + length):
            for j in range(start_y, start_y + length):
                pic[i,j] = np.random.randint(min, max)

        self.pic = pic

        return pic



    def gauss_start(self, average, sigma):
        """
        Adds normal noise on image
        param:
        average - average of normal distribution
        sigma - sigma of normal distribution

        ***
        Changes self.pic array to self.pic with normal noise
        """
        for i in range(0,self.width):
            for j in range(0,self.height):
                self.pic[i][j] = np.random.normal(average, sigma) #первое число  - среднее, второе число  - стандартное отклонение


    def add_round_shape_gauss(self, radius, center_x, center_y, average, sigma):
        """
        Adds circle with gauss noise on image
        param:
        radius - radius of circle
        center_x - coord. x of center
        center_y - coord. y of center
        avarage, sigma  - params of normal distribution
        """
        x = radius
        y = 0
        error = 1 - x

        while x >= y:

            self.pic[int(center_y + y)][int(center_x + x)] = np.random.normal(average, sigma)
            self.pic[int(center_y + x)][int(center_x + y)] = np.random.normal(average, sigma)
            self.pic[int(center_y - y)][int(center_x + x)] = np.random.normal(average, sigma)
            self.pic[int(center_y + x)][int(center_x - y)] = np.random.normal(average, sigma)
            self.pic[int(center_y - y)][int(center_x - x)] = np.random.normal(average, sigma)
            self.pic[int(center_y - x)][int(center_x - y)] = np.random.normal(average, sigma)
            self.pic[int(center_y + y)][int(center_x - x)] = np.random.normal(average, sigma)
            self.pic[int(center_y - x)][int(center_x + y)] = np.random.normal(average, sigma)

            y += 1
            if error < 0:
                error += 2 * y + 1
            else:
                x -= 1
                error += 2 * (y - x + 1)

        return self.pic


    def add_round_shape(self, radius, center_x, center_y, max, min):
        """
        Adds circle with random.randint() distribution
        param:
        radius - radius of circle
        center_x - coord. x of center
        center_y - coord. y of center
        max, min  - params of random.randint() distribution
        """
        x = radius
        y = 0
        error = 1 - x

        while x >= y:
            self.pic[int(center_y + y)][int(center_x + x)] = np.random.randint(min, max)
            self.pic[int(center_y + x)][int(center_x + y)] = np.random.randint(min, max)
            self.pic[int(center_y - y)][int(center_x + x)] = np.random.randint(min, max)
            self.pic[int(center_y + x)][int(center_x - y)] = np.random.randint(min, max)
            self.pic[int(center_y - y)][int(center_x - x)] = np.random.randint(min, max)
            self.pic[int(center_y - x)][int(center_x - y)] = np.random.randint(min, max)
            self.pic[int(center_y + y)][int(center_x - x)] = np.random.randint(min, max)
            self.pic[int(center_y - x)][int(center_x + y)] = np.random.randint(min, max)

            y += 1
            if error < 0:
                error += 2 * y + 1
            else:
                x -= 1
                error += 2 * (y - x + 1)



    def view(self):
        """
        Shows image on screen
        param:
        """
        for i in range(0, self.width):
            for j in range(0, self.height):
                if (self.pic[i][j] != 0):
                    self.draw.point((i, j), fill=int(self.pic[i][j]))
                else:
                    color = 0
                    self.draw.point((i, j), fill=color)
                    pic[i][j] = color
        self.image.show()




    def save(self, name, path):
        """
        Saves image
        param:
        name - name of file
        path - path to file where user wants to save image
        """
        self.pic = np.nan_to_num(self.pic)
        color = 0
        for i in range(0, self.width):
            for j in range(0, self.height):
                #print(round(self.pic[i][j]))
                if(self.pic[i][j] < 1):
                    color = 0
                else:
                    color = int(self.pic[i][j])
                    self.draw.point((i, j), fill=color)
        file_name = name
        self.image.save(f'{path}\{file_name}.png')


    def im_show(self):
        """
        returns image object from PIL that is made from self.pic array
        param:
        returns:
        image - PIL object

        """
        for i in range(0, 8):
            for j in range(0, 8):
                if(self.pic[i][j] < 1):
                    color = 0
                else:
                    color = int(self.pic[i][j])
                    self.draw.point((i, j), fill=color)
        return self.image



"Structure class - determines the structure, its luminescence, the change in its conductivity with time"
class Structure(object):
    C1 = 1.2
    C2 = 1.2
    tau_1 = 0.14
    tau_2 = 1.35
    k = 0.1


    def __init__(self, name, x, y, image, pow, coef, time):
        self.name = name
        self.coord_x = x
        self.coord_y = y
        self.image = image
        self.pow = pow
        self.coef = coef
        self.color = self.image.getpixel((x, y))
        self.Conductivity = self.color
        self.time0 = time
        self.time = np.arange(0, self.time0, 1)
        self.event_array = np.zeros(len(self.time))
        self.file1 = r'/Users/tortolla/Desktop/model/calibration.txt'
        self.file2 = r'/Users/tortolla/Desktop/model/calibration_relax.txt'
        self.cond = np.array(())
        self.impulse_time = 30
        self.x1 = np.array(())
        self.x2 = np.array(())
        self.x3 = np.array(())
        self.x4 = np.array(())
        self.array1_imp, self.array2_imp = read_arrays_from_file(self.file1, 1)
        self.array1_relax, self.array2_relax = read_arrays_from_file(self.file2, 0)

        p1 = 255 / self.array1_imp[-1]
        p2 = 255 / self.array1_relax[0]

        self.array1_imp = self.array1_imp * p1
        self.array1_relax = self.array1_relax * p2


        # with open(r"C:\Users\ALIENWARE\PycharmProjects\model\venv\модель\aprox.csv", "r") as file:
        #     reader = csv.reader(file)
        #
        #     # Пропускаем первую строку
        #     for row in reader:
        #         values = row[0].split()  # Разделяем строку на отдельные значения
        #
        #         try:
        #             self.cond = np.append(self.cond, float(values[0]))
        #         except ValueError:
        #             self.cond = np.append(self.cond, None)
        #
        #         try:
        #             self.x1 = np.append(self.x1, float(values[1]))
        #         except ValueError:
        #             self.x1 = np.append(self.x1, None)
        #
        #         try:
        #             self.x2 = np.append(self.x2, float(values[2]))
        #         except ValueError:
        #             self.x2 = np.append(self.x2, None)
        #
        #         try:
        #             self.x3 = np.append(self.x3, float(values[3]))
        #         except ValueError:
        #             self.x3 = np.append(self.x3, None)
        #
        #         try:
        #             self.x4 = np.append(self.x4, float(values[4]))
        #         except ValueError:
        #             self.x4 = np.append(self.x4, None)


    def Poisson_generator(self, rate, n, myseed=False):
        """
        The Poisson_generator function is used to generate a Poisson process with a given frequency (rate) and duration (n) to simulate random events
        param:
        rate - frequancy of Poisson process
        n - duration of the Poisson process
        myseed - initial value for the random number generator
        returns:
        poisson train - array with 0/1 for time duration
        """


        range_t = self.time
        Lt = range_t.size


        if myseed:
            np.random.seed(seed=myseed)
        else:
            np.random.seed()


        u_rand = np.random.rand(n, Lt)

        firing_delimeter = 4
        dt = 1
        poisson_train = 1. * (u_rand < rate * (dt / firing_delimeter))
        poisson_train = poisson_train.flatten()
        count  = 0
        for i in poisson_train:
            if i == 1:
                count +=1

        return poisson_train



    def encode_pixel_brightness(self):
        """
        Translates pixel brightness into rate param for Poisson generator
        returns:
        spike_train - array with 0/1 for time duration (where 1 is spike, and 0 is relax)
        """

        max_brightness = 255
        rate = pow((self.color/(max_brightness)), self.pow) * self.coef
        #rate = pow((self.color / (max_brightness)), 6) / 2
        n = 1
        spike_train = self.Poisson_generator(rate, n)
        spike_list = [i for i, spike in enumerate(spike_train.tolist())]


        return spike_train



    def get_self_event_array(self):
        """
        Function that gets self_event_array
        returns
        event_array -  array with 0/1 for time duration (where 1 is spike, and 0 is relax)
        """

        self.event_array = self.encode_pixel_brightness()

        return self.event_array





    def sgo_event(self):
        """
        Function that counts the amount of spikes in self.event_array
        returns:
        num - number of 1( number of spikes ) in event_array
        """

        num = 0
        min_len = 0
        flag = 0
        z = 0
        numer = 0
        delta = 0
        event_array = self.event_array[:6000]
        for i in event_array:
            if(i == 1):
                num += 1
            numer+=1
        return(num)


    def impulse(self, conductivity):
        """
        Impulse function that counts conductivity after one spike ( after 1 in event_array )
        param:
        conductivity - conductivity of structure before spike ( before 1 in event_array )
        returns:
        conductivity - conductivity of structure after spike ( after 1 in event_array )
        """

        time_index = find_closest_value(conductivity, self.array1_imp)
        dt = self.array2_imp[time_index]

        A = -4.90752341e+00
        B = 1.61927247e+00
        C = -3.46762146e-03
        D = 2.71757824e-06

        up = A + B*(dt + self.impulse_time) + C * pow(dt + self.impulse_time, 2) + D * pow(dt + self.impulse_time, 3)
        if(up <= 255):
            return up
        else:
            return 255


    def relax(self, conductivity, time):
       """
       Relaxation function that counts conductivity after one relax event( after 0 in event_array )
       param:
       conductivity - conductivity of structure before spike ( before 0 in event_array )
       returns:
       conductivity - conductivity of structure after spike ( after 0 in event_array )
       """

       time_index = find_closest_value(conductivity, self.array1_relax)
       time = time * self.impulse_time

       dt = self.array2_relax[time_index]

       A = 2.33142078e+02
       B = 2.96052741e-04
       C = 3.96503308e+01
       D = 2.96035007e-04

       down = (A * np.exp(B * (-(time+dt))) + C * np.exp(D * (-(time+dt))))

       if(down > 0):
           return down
       else:
           return 0.0000000001


    def get_sequence_length(self, start_index, target_value):
        """
        Gets length of sequance from start_index of array until target value in array
        param:
        start_index - index from that sequence starts
        target_value - value that we are searching in array
        returns:
        length - length of seqeunce
        """
        length = 0
        for i in range(start_index, len(self.event_array)):
            if self.event_array[i] == target_value:
                length += 1
            else:
                break
        return length


    def cond_in_time(self, time):
        """
        Function that calculates structures conductivity in time
        param:
        time - time, when user whants to now conductivity
        returns:
        cond - conductivity in time
        """

        i = 0
        cond = self.color
        event_array = self.event_array[:time]

        "Take the event_array before the desired time"
        while (i < (len(event_array))):

            "If 1, we simply apply the pulse at once"
            if (event_array[i] == 1):
                    cond = self.impulse(cond)
                    i+=1
            else:
                "Have to count how long it takes to relax"

                relax_time = 0

                while (event_array[i] == 0):
                    "If the relaxation time is less than the length of the event_array - just look at the next element and add one to the relaxation time"

                    if ((i + relax_time) < (len(event_array) - 1)):

                        relax_time += 1
                        i += 1

                    else:
                        i+=1
                        break
                if(cond == 'None'):
                    cond  = 0
                cond  = self.relax(cond, relax_time)
                #i += length

        return cond









