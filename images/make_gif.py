import imageio

NAME = "part_6"
NUM_FRAMES = 20
FPS = 15

imgs = []
for frame in range(NUM_FRAMES):
    fname = f"{NAME}/{NAME}_{frame}.png"
    img = imageio.imread(fname)
    imgs.append(img)
imageio.mimwrite(f"{NAME}.gif", imgs, fps = FPS, loop = 0)