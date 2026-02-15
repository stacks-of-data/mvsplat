from pathlib import Path

import numpy as np
import torch
from einops import einsum, rearrange
from jaxtyping import Float
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R
from torch import Tensor


def construct_list_of_attributes(num_rest: int) -> list[str]:
    attributes = ["x", "y", "z", "nx", "ny", "nz"]
    # Standard SH fields
    for i in range(3):
        attributes.append(f"f_dc_{i}")
    for i in range(num_rest):
        attributes.append(f"f_rest_{i}")
    attributes.append("opacity")
    # Scaling and Rotation
    for i in range(3):
        attributes.append(f"scale_{i}")
    for i in range(4):
        attributes.append(f"rot_{i}")
    # Standard RGB fields for general viewers
    attributes.append("red")
    attributes.append("green")
    attributes.append("blue")
    return attributes

def SH2RGB(sh):
    return sh * 0.28209479177387814 + 0.5

def export_ply(
    extrinsics: Float[Tensor, "4 4"],
    means: Float[Tensor, "gaussian 3"],
    scales: Float[Tensor, "gaussian 3"],
    rotations: Float[Tensor, "gaussian 4"],
    harmonics: Float[Tensor, "gaussian 3 d_sh"],
    opacities: Float[Tensor, " gaussian"],
    path: Path,
):
    # SH DC coefficient constant
    C0 = 0.28209479177387814
    
    # Shift the scene so that the median Gaussian is at the origin.
    means = means - means.median(dim=0).values

    # Rescale the scene so that most Gaussians are within range [-1, 1].
    scale_factor = means.abs().quantile(0.95, dim=0).max()
    means = means / scale_factor
    scales = scales / scale_factor

    # Define a rotation that makes +Z be the world up vector.
    rotation = [
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0],
    ]
    rotation = torch.tensor(rotation, dtype=torch.float32, device=means.device)

    # The Polycam viewer seems to start at a 45 degree angle. Since we want to be
    # looking directly at the object, we compose a 45 degree rotation onto the above
    # rotation.
    adjustment = torch.tensor(
        R.from_rotvec([0, 0, -45], True).as_matrix(),
        dtype=torch.float32,
        device=means.device,
    )
    rotation = adjustment @ rotation

    # We also want to see the scene in camera space (as the default view). We therefore
    # compose the w2c rotation onto the above rotation.
    rotation = rotation @ extrinsics[:3, :3].inverse()

    # Apply the rotation to the means (Gaussian positions).
    means = einsum(rotation, means, "i j, ... j -> ... i")

    # Apply the rotation to the Gaussian rotations.
    rotations = R.from_quat(rotations.detach().cpu().numpy()).as_matrix()
    rotations = rotation.detach().cpu().numpy() @ rotations
    rotations = R.from_matrix(rotations).as_quat()
    x, y, z, w = rearrange(rotations, "g xyzw -> xyzw g")
    rotations = np.stack((w, x, y, z), axis=-1)

    # Prepare Spherical Harmonics (DC Component)
    # harmonics shape is [G, 3, d_sh]. We want the first coefficient for each channel.
    # Note: Some models store SH as [G, d_sh, 3], adjust if necessary.
    sh_dc = harmonics[..., 0]

    # Prepare Opacity (applying sigmoid if your model outputs raw logits)
    # opacities = torch.sigmoid(opacities) # Uncomment if model outputs logits
    opacity = opacities[..., None].detach().cpu().numpy()

    colors_rgb = SH2RGB(sh_dc).detach().cpu().numpy()
    colors_rgb = (np.clip(colors_rgb, 0, 1) * 255).astype(np.uint8)

    attribute_names = construct_list_of_attributes(0)
    
    # Coordinates, SH, Opacity, Scale, Rot are float32 ('f4')
    # Red, Green, Blue are uint8 ('u1')
    v_list = [(name, "f4") for name in attribute_names[:-3]]
    v_list += [(name, "u1") for name in attribute_names[-3:]]
    
    elements = np.empty(means.shape[0], dtype=v_list)

    # 4. Concatenate and convert to list of tuples
    # We keep the floats and uint8s separate for the stack, 
    # then map them into the structured array
    attributes_float = np.concatenate([
        means.detach().cpu().numpy(),
        torch.zeros_like(means).detach().cpu().numpy(), # normals
        sh_dc.detach().cpu().numpy(),
        opacity,
        scales.log().detach().cpu().numpy(),
        rotations,
    ], axis=1)

    # Fill the structured array
    for i, name in enumerate(attribute_names[:-3]):
        elements[name] = attributes_float[:, i]
    
    elements["red"] = colors_rgb[:, 0]
    elements["green"] = colors_rgb[:, 1]
    elements["blue"] = colors_rgb[:, 2]

    # 5. Save
    path.parent.mkdir(exist_ok=True, parents=True)
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)
