import skimage.io

fname = '/home/cindy/PycharmProjects/data/LOL/eval15_256/low_778'

for i in range(2, 11):
    img = skimage.io.imread(f'{fname}/1.png')
    skimage.io.imsave(f'{fname}/{i}.png', img)