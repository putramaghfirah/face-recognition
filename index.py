# mengimport library yang digunakan
import face_recognition
import copy
from pathlib import Path
from PIL import Image

# mengambil tiap foto yang ada di dalam direktori img
paths = [path for path in Path('img').glob('*.jpg')]

# kernel untuk blur
blur_kernel = [
    [1/16, 1/8, 1/16],
    [1/8, 1/4, 1/8],
    [1/16, 1/8, 1/16]
]

weight = 0.04

# kernel untuk sharp
sharpen_kernel = [
    [0, -1 * weight, 0, ],
    [-1 * weight, 4*weight + 1, -1 * weight],
    [0, -1 * weight, 0, ]
]

# black kernel
black_kernel = [
    [0.05, 0.05, 0.05],
    [0.05, 0.05, 0.05],
    [0.05, 0.05, 0.05]
]


# fungsi untuk menfilter image dengan paramater foto, lokasi wajah, dan kernel dari filter
def filter_image(image, face_location, kernel: list):
    # membagi lokasi wajah ke dalam top, right, bottom, left
    top, right, bottom, left = face_location
    # melakukan looping untuk vertikal dari top ke bottom
    for y in range(top, bottom):
        # melakukan looping untuk horizontal dari left ke right
        for x in range(left, right):
            # melakukan terhadap layer dari sebuah foto, dikarenakan dari tiap foto terdapat 3 layer, yaitu r,g,b
            for i in range(3):
                # variabel total untuk menyimpan hasil perkalian kernel dengan pixel-pixel tiap foto
                total = 0
                # membuat skala 3x3
                for vertikal in range(-1, 2):
                    for horizontal in range(-1, 2):
                        # menjumlahkan total dengan image dikali dengan kernel
                        total += (image[vertikal+y][horizontal+x][i] *
                                  kernel[vertikal+1][horizontal+1])
                # hasil dari total dimasukkan ke dalam tiap pixel image
                image[y][x][i] = total
    # lalu mengembalikan hasil dari image
    return image


# melakukan looping untuk tiap foto yang telah didapat dari direktori/folder img
for path in paths:
    print(f"Memfilter foto {path}")
    # mengambil nama file dari tiap foto lalu memasukkan ke dalam variable file_name
    file_name = path.name
    # dari foto yang ada diubah ke dalam numpy array/list
    image = face_recognition.load_image_file(path)
    # mengcopy tiap foto agar tidak menimpa hasil filternya
    image_blur = copy.deepcopy(image)
    image_sharp = copy.deepcopy(image)
    image_black = copy.deepcopy(image)
    # mengambil wajah yang ada dari tiap foto
    face_locations = face_recognition.face_locations(image)
    print(f"Lokasi wajah : {face_locations}")
    # melakukan filter terhadap wajah yang didapat dari sebuah foto
    for face_location in face_locations:
        # for i in range(5):
        # melakukan filter blur/sharp/black
        blur = filter_image(image_blur, face_location, blur_kernel)
        sharp = filter_image(image_sharp, face_location, sharpen_kernel)
        black = filter_image(image_black, face_location, black_kernel)

    # menyimpan hasil filter ke dalam pil_image lalu save foto yang telah difilter
    pil_image_blur = Image.fromarray(blur)
    pil_image_blur.save(f"out/blurred_{file_name}")
    pil_image_sharp = Image.fromarray(sharp)
    pil_image_sharp.save(f"out/sharped_{file_name}")
    pil_image_black = Image.fromarray(black)
    pil_image_black.save(f"out/black_{file_name}")
    print(f"foto {path.name} telah difilter")
