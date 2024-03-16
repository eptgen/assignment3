import imageio

NAME = "part_7_12"
OUT_NAME = f"../../proj3/data/{NAME}_geometry.gif"
NUM_FRAMES = 20
FPS = 10

imgs = []
for frame in range(NUM_FRAMES):
    fname = f"{NAME}/{NAME}_{frame}.png"
    img = imageio.imread(fname)
    imgs.append(img)
imageio.mimwrite(OUT_NAME, imgs, fps = FPS, loop = 0)