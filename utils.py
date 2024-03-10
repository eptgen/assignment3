import numpy as np
import torch
from pytorch3d.io import load_obj
from pytorch3d.ops import cubify
from pytorch3d.renderer import (
    AlphaCompositor,
    FoVPerspectiveCameras,
    HardPhongShader,
    MeshRenderer,
    MeshRasterizer,
    PointLights,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    RasterizationSettings,
    TexturesVertex,
)
from pytorch3d.renderer.cameras import look_at_view_transform
from pytorch3d.structures import Pointclouds


def get_device():
    """
    Checks if GPU is available and returns device accordingly.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device


def get_points_renderer(
    image_size=512, device=None, radius=0.01, background_color=(1, 1, 1)
):
    """
    Returns a Pytorch3D renderer for point clouds.

    Args:
        image_size (int): The rendered image size.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
        radius (float): The radius of the rendered point in NDC.
        background_color (tuple): The background color of the rendered image.
    
    Returns:
        PointsRenderer.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = PointsRasterizationSettings(image_size=image_size, radius=radius,)
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )
    return renderer


def get_mesh_renderer(image_size=512, lights=None, device=None):
    """
    Returns a Pytorch3D Mesh Renderer.

    Args:
        image_size (int): The rendered image size.
        lights: A default Pytorch3D lights object.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = RasterizationSettings(
        image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=HardPhongShader(device=device, lights=lights),
    )
    return renderer


def unproject_depth_image(image, mask, depth, camera):
    """
    Unprojects a depth image into a 3D point cloud.

    Args:
        image (torch.Tensor): A square image to unproject (S, S, 3).
        mask (torch.Tensor): A binary mask for the image (S, S).
        depth (torch.Tensor): The depth map of the image (S, S).
        camera: The Pytorch3D camera to render the image.
    
    Returns:
        points (torch.Tensor): The 3D points of the unprojected image (N, 3).
        rgba (torch.Tensor): The rgba color values corresponding to the unprojected
            points (N, 4).
    """
    device = camera.device
    assert image.shape[0] == image.shape[1], "Image must be square."
    image_shape = image.shape[0]
    ndc_pixel_coordinates = torch.linspace(1, -1, image_shape)
    Y, X = torch.meshgrid(ndc_pixel_coordinates, ndc_pixel_coordinates)
    xy_depth = torch.dstack([X, Y, depth])
    points = camera.unproject_points(
        xy_depth.to(device), in_ndc=False, from_ndc=False, world_coordinates=True,
    )
    points = points[mask > 0.5]
    rgb = image[mask > 0.5]
    rgb = rgb.to(device)

    # For some reason, the Pytorch3D compositor does not apply a background color
    # unless the pointcloud is RGBA.
    alpha = torch.ones_like(rgb)[..., :1]
    rgb = torch.cat([rgb, alpha], dim=1)

    return points, rgb


def load_cow_mesh(path="data/cow_mesh.obj"):
    """
    Loads vertices and faces from an obj file.

    Returns:
        vertices (torch.Tensor): The vertices of the mesh (N_v, 3).
        faces (torch.Tensor): The faces of the mesh (N_f, 3).
    """
    vertices, faces, _ = load_obj(path)
    faces = faces.verts_idx
    return vertices, faces
    
def render_gif(renderer, model, num_povs, device, camera = None):
    rends = []
    for i in range(num_povs):
        theta = 360 * i * (1 / num_povs)
        cameras = camera
        if cameras is None:
            R, T = look_at_view_transform(dist = 2., azim = theta)
            cameras = FoVPerspectiveCameras(
                R=R, T=T, fov=60, device = device
            )
        lights = PointLights(location=[[0, 0, -3]], device = device)
        rend = renderer(model, cameras=cameras, lights=lights)
        rends.append((rend.detach().cpu().numpy()[0, ..., :3] * 255).astype(np.uint8))
    return rends
    
def render_from_angle(mesh, angle, args):
    color = torch.tensor([0.7, 0.7, 1], device = args.device)
    renderer = get_mesh_renderer(image_size=256)
    mesh_textures = torch.ones_like(mesh.verts_packed(), device = args.device)
    mesh_textures = mesh_textures * color
    mesh.textures = TexturesVertex(mesh_textures.unsqueeze(0))
    R, T = look_at_view_transform(dist = 2., azim = angle)
    cameras = FoVPerspectiveCameras(
        R=R, T=T, fov=60, device=args.device
    )
    lights = PointLights(location=[[0, 0, -3]], device=args.device)
    rend = renderer(mesh, cameras=cameras, lights=lights)
    return (rend.detach().cpu().numpy()[0, ..., :3] * 255).astype(np.uint8)

def render_voxel(voxels, args):
    color = torch.tensor([0.7, 0.7, 1], device = args.device)
    
    renderer = get_mesh_renderer(image_size=256)
    mesh = cubify(voxels, 0.5)
    mesh = mesh.to(args.device)
    mesh_textures = torch.ones_like(mesh.verts_packed(), device = args.device)
    mesh_textures = mesh_textures * color
    mesh.textures = TexturesVertex(mesh_textures.unsqueeze(0))
    return render_gif(renderer, mesh, args, 10)

def render_one(renderer, model, device, camera = None):
    return render_gif(renderer, model, 1, device, camera)[0]

def render_cloud(points, output_gif, device, camera = None):
    color = torch.tensor([0.7, 0.7, 1.0], device = device)
    renderer = get_points_renderer(image_size=256, device = device)
    rgb = torch.ones_like(points, device = device) * color
    pc = Pointclouds(
        points=points.unsqueeze(0),
        features=rgb.unsqueeze(0),
    ).to(device)
    
    if output_gif: return render_gif(renderer, pc, 10, device)
    return render_one(renderer, pc, device, camera)
    
def render_mesh(mesh, args):
    color = torch.tensor([0.7, 0.7, 1.0], device = args.device)
    
    renderer = get_mesh_renderer(image_size=256)
    mesh_textures = torch.ones_like(mesh.verts_packed(), device = args.device)
    mesh_textures = mesh_textures * color
    mesh.textures = TexturesVertex(mesh_textures.unsqueeze(0))
    
    return render_gif(renderer, mesh, args, 10)