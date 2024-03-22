import torch
import trimesh
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex
)


def render_d(vertices: torch.Tensor, faces: torch.Tensor, image_size=(256, 256)):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    R, T = look_at_view_transform(eye=torch.tensor([[0, 0, 250]], dtype=torch.float32),
                                  up=((0, 1, 0),),
                                  at=((0, 0, 0),),
                                  device=device)

    # 创建相机
    cameras = FoVPerspectiveCameras(
        device=device, R=R, T=T,
        znear=0.1, zfar=1000,
        fov=50,
    )

    # 渲染设置
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    # 光源
    lights = PointLights(device=device, location=[[0.0, 0.0, 100.0]])

    # mesh
    p3_mesh = Meshes([vertices],
                     [faces],
                     textures=TexturesVertex(verts_features=[torch.ones(vertices.shape, device=device)]))

    # 创建渲染器
    renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
                            shader=SoftPhongShader(device=device, cameras=cameras, lights=lights))

    # 渲染图像
    images = renderer(p3_mesh)
    images = images[0, ..., :3]

    return images
