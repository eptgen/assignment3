import imageio

NAME = "part_7"
OUT_NAME = f"{NAME}_geometry.gif"
NUM_FRAMES = 20
FPS = 15

imgs = []
for frame in range(NUM_FRAMES):
    fname = f"{NAME}/{NAME}_{frame}.png"
    img = imageio.imread(fname)
    imgs.append(img)
imageio.mimwrite(OUT_NAME, imgs, fps = FPS, loop = 0)