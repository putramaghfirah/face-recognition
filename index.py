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


def blur(image, x, y):
    for i in range(3):
        total = 0
        for vertikal in range(-1, 2):
            for horizontal in range(-1, 2):
                total += (image[vertikal+y][horizontal+x][i] *
                          blur_kernel[vertikal+1][horizontal+1])
        image[y][x][i] = total
    return image


def filter_image(image, face_location, method):
    top, right, bottom, left = face_location
    for y in range(top, bottom):
        for x in range(left, right):
            image = method(image, x, y)
    return image


for path in paths:
    image = face_recognition.load_image_file(path)
    face_locations = face_recognition.face_locations(image)
    for face_location in face_locations:
        for i in range(25):
            image = filter_image(image, face_location, blur)
    pil_image = Image.fromarray(image)
    pil_image.save("out/tes.jpg")
