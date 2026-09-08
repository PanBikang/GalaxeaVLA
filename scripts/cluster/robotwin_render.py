"""Exercise real ray-traced image acquisition on the allocated CUDA GPU."""
import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import sapien
from sapien import render

assert torch.cuda.is_available()
device = sapien.Device('cuda:0')
assert device.cuda_id == 0
print('CUDA:', torch.cuda.get_device_name(), 'SAPIEN:', device, 'PCI:', device.pci_string, flush=True)
render.set_global_config(max_num_materials=50000, max_num_textures=50000)
render.set_camera_shader_dir('rt')
render.set_ray_tracing_samples_per_pixel(32)
render.set_ray_tracing_path_depth(8)
render.set_ray_tracing_denoiser('oidn')
scene = sapien.Scene([sapien.physx.PhysxCpuSystem(), render.RenderSystem(device)])
scene.set_ambient_light([0.5,0.5,0.5])
scene.add_directional_light([1,1,-1],[1,1,1])
builder = scene.create_actor_builder()
builder.add_box_visual(half_size=[0.15]*3, material=render.RenderMaterial(base_color=[0.8,0.1,0.1,1]))
actor = builder.build_static(name='box')
actor.set_pose(sapien.Pose([0,0,0.25]))
camera = scene.add_camera('check',128,128,1.0,0.01,10)
camera.set_pose(sapien.Pose([-1,0,0.25]))
scene.update_render()
camera.take_picture()
rgba = camera.get_picture('Color')
assert rgba.shape == (128,128,4) and np.isfinite(rgba).all()
assert float(rgba[:,:,:3].std()) > 0.01
path = Path(os.environ['PROJECT_ROOT']) / 'runs/robotwin/provenance' / f'render-{os.environ["SLURM_JOB_ID"]}.png'
path.parent.mkdir(parents=True,exist_ok=True)
Image.fromarray((np.clip(rgba[:,:,:3],0,1)*255).astype(np.uint8)).save(path)
print('RAY-TRACED RENDER PASS',path,flush=True)
