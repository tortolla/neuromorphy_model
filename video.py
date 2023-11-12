import cv2 as cv2
import os


image_folder = r'модель' #folder with all images
video_name = r'video.avi' #name of feature video


"""
This part is only to check the work of openCV
"""


image_folder = r'C:\Users\ALIENWARE\PycharmProjects\model\venv\модель' #path to image folder

img = cv2.imread(r"C:\\Users\\ALIENWARE\\PycharmProjects\\model\\venv\\импульс.png") #shows one image

print(os.path.exists(r'C:\\Users\\ALIENWARE\\PycharmProjects\\model\\venv\\импульс.png')) #if TRUE(1) than path is valid, else - path is invalid

cv2.imshow("Display window", img)
k = cv2.waitKey(0)


"""
This part actually makes video from users images
"""


images = [img for img in os.listdir(image_folder) if img.endswith(".png")] #takes all names of images in folder to the 'str' array

print(images) #print their names to check if everything works fine

"""
Taking image parameters to understand future video parameters
"""


frame = cv2.imread(os.path.join(image_folder, images[0]))

height, width, layers = frame.shape


"""
Making video 
"""

video = cv2.VideoWriter(video_name, 0, 1, (width, height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()

video.release()
