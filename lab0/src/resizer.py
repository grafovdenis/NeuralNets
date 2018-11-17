# This program will transform image to smaller resolution and converts its to greyscale

from PIL import Image  # PLEASE INSTALL PILLOW, NOT JUST PIL
import matplotlib.pyplot as plt
import os

# they must be existing
directory = '../images/'
output_dir = '../output'

resolution = (64, 64)

for filename in os.listdir(directory):
    img = Image.open(directory + filename).convert('L')  # flag 'L' converts to greyscale
    img.thumbnail(resolution, Image.ANTIALIAS)
    fig = plt.imshow(img)
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    # plt.show()
    plt.savefig(os.path.join(output_dir, filename.split('.')[0] + '.png'), bbox_inches='tight', pad_inches=0)
