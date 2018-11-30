# This program will transform directory of images to smaller resolution and converts its to greyscale.
# Example:
# input_directory = '../input/'
# output_directory = '../output/'
# resolution = (64, 64)
# transform_directory(input_directory, output_directory, resolution)
# transform_to_single_file('../input/', 'output.png', (128, 128))

from PIL import Image  # PLEASE INSTALL PILLOW, NOT JUST PIL
import matplotlib.pyplot as plt
import os
import numpy as np


def centered_crop(img, new_height, new_width):
    width, height = img.size  # Get dimensions
    if width == height:
        return img
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    return img.crop((left, top, right, bottom))


def transform_directory(input_directory, output_directory, resolution):
    if not os.path.exists(input_directory):
        print("Directory not exists")
        return
    for filename in os.listdir(input_directory):
        img = Image.open(input_directory + filename).convert('L')  # flag 'L' converts to greyscale
        img.thumbnail(resolution, Image.ANTIALIAS)
        fig = plt.imshow(img)
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        # plt.show()
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        plt.savefig(os.path.join(output_directory, filename.split('.')[0] + '.png'), bbox_inches='tight', pad_inches=0)


# TODO Исправить центрирование
def transform_to_single_file(input_directory, output_file, resolution):
    if not os.path.exists(input_directory):
        print("Directory not exists")
        return

    dir_len = len(os.listdir(input_directory))
    rows = int(np.sqrt(dir_len))
    new_im = Image.new('RGB', (resolution[0] * rows, resolution[1] * rows))
    cur_row = 0
    for index, filename in enumerate(os.listdir(input_directory)):
        cur_column = index - cur_row * rows
        img = Image.open(input_directory + filename).convert('L')  # flag 'L' converts to greyscale
        img.thumbnail((resolution[0] * 1.5, resolution[1] * 1.5), Image.ANTIALIAS)
        img = centered_crop(img, resolution[0], resolution[1])
        new_im.paste(img, (cur_column * resolution[0], cur_row * resolution[1]))
        if (index + 1) % rows == 0:
            cur_row += 1
    new_im.show()
    new_im.save(os.path.join('../output/', output_file + '.png'))

