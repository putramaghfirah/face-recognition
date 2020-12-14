import face_recognition
from pathlib import Path
from PIL import Image

paths = [path for path in Path('img').glob('*.jpg')]

blur_kernel = [
    [1/16, 1/8, 1/16],
    [1/8, 1/4, 1/8],
    [1/16, 1/8, 1/16]
]

weight = 0.02

sharpen_kernel = [
    [0, -1 * weight, 0, ],
    [-1 * weight, 4*weight + 1, -1 * weight],
    [0, -1 * weight, 0, ]
]


def filter_image(image, face_location, kernel: list):
    top, right, bottom, left = face_location
    for y in range(top, bottom):
        for x in range(left, right):
            for i in range(3):
                total = 0
                for vertikal in range(-1, 2):
                    for horizontal in range(-1, 2):
                        total += (image[vertikal+y][horizontal+x][i] *
                                  kernel[vertikal+1][horizontal+1])
                image[y][x][i] = total
    return image


for path in paths:
    file_name = path.name
    image = face_recognition.load_image_file(path)
    face_locations = face_recognition.face_locations(image)
    for face_location in face_locations:
        for i in range(15):
            # image_blur = filter_image(image, face_location, blur_kernel)
            image_sharp = filter_image(image, face_location, sharpen_kernel)
    # pil_image_blur = Image.fromarray(image_blur)
    # pil_image_blur.save(f"out/blurred_{file_name}")
    pil_image_sharp = Image.fromarray(image_sharp)
    pil_image_sharp.save(f"out/sharped_{file_name}")
