import face_recognition
from pathlib import Path
from PIL import Image

paths = [path for path in Path('img').glob('*.jpg')]

blur_kernel = [
    [1/16, 1/8, 1/16],
    [1/8, 1/4, 1/8],
    [1/16, 1/8, 1/16]
]

sharpen_kernel = [
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
]


# def blur(image, x, y):
#     for i in range(3):
#         total = 0
#         for vertikal in range(-1, 2):
#             for horizontal in range(-1, 2):
#                 total += (image[vertikal+y][horizontal+x][i] *
#                           blur_kernel[vertikal+1][horizontal+1])
#         image[y][x][i] = total
#     return image


# def sharpen(image, x, y):
#     for i in range(3):
#         total = 0
#         for vertikal in range(-1, 2):
#             for horizontal in range(-1, 2):
#                 total += (image[vertikal+y][horizontal+x][i] *
#                           sharpen_kernel[vertikal+1][horizontal+1])
#         image[y][x][i] = total
#     return image

def blur(image, x, y):
    skala = 3
    padding = int(skala/2)
    for i in range(3):
        total = 0
        for vertikal in range(0-padding, padding+1):
            for horizontal in range(0-padding, padding+1):
                total += image[vertikal+y][horizontal+x][i]
        mean = total/(skala*skala)
        # mean = (int(image[y + 1][x - 1][i]) +
        #         int(image[y + 1][x][i]) +
        #         int(image[y + 1][x + 1][i]) +
        #         int(image[y][x - 1][i]) +
        #         int(image[y][x][i]) +
        #         int(image[y][x + 1][i]) +
        #         int(image[y - 1][x - 1][i]) +
        #         int(image[y - 1][x][i]) +
        #         int(image[y - 1][x + 1][i])) / 9
        image[y][x][i] = mean


def filter_image(image, face_location):
    top, right, bottom, left = face_location
    for y in range(top, bottom):
        for x in range(left, right):
            image = blur(image, x, y)
    return image


for path in paths:
    image = face_recognition.load_image_file(path)
    face_locations = face_recognition.face_locations(image)
    for face_location in face_locations:
        image = filter_image(image, face_location)
    pil_image = Image.fromarray(image)
    pil_image.save("out/tes.jpg")
