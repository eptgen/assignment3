
In order to get the depth_map, run the command `python volume_rendering_main.py --config-name=box` and the depth map will appear in `images/depth_map.png`. Additionally, the folder `images/part_1` will contain frames of the volumetric render.

To train the volumetric model, run the command `python volume_rendering_main.py --config-name=train_box`. The GIF and images will appear in the `images` directory, and the box center and side lengths will be outputted.

To train the NeRF, run the command `python volume_rendering_main.py --config-name=nerf_lego`. The GIF will appear in the `image` directory. The `implicit.py` file has code for both view-dependent and view-independent training methods, and one can comment out the relevant parts of the code (guided by the comments) to use the desired training method.


