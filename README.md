# ERGO

Description.

## Getting Started

Please download a pretrained motionBERT file from [HERE](https://onedrive.live.com/?authkey=%21ABOq3JHlmyCLz9k&id=A5438CD242871DF0%21170&cid=A5438CD242871DF0). Please rename the file as `MB_best_epoch.bin` and place it into `models/pt`.  
```
git clone https://github.com/gwyong/ergo.git
pip install -r requirements.txt
```

### 2D/3D Pose Estimation from Image
```python
import os
import numpy as np
import torch
import torch.nn as nn

from functools import partial
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from ultralytics import YOLO

import utils
from data.utils import coco2h36m, draw_2d_pose_from_image
from models.motionBERT import DSTformer, get_config, render_and_save

device = 'cuda' if torch.cuda.is_available() else 'cpu'

image = "worker.jpg"
image_filename = utils.get_filename(image)

pose2d_model = YOLO('yolov8n-pose.pt')
keypoints = pose2d_model(image)[0].keypoints.cpu().numpy() # shape (num_humans, 17, 3), COCO format
keypoints = coco2h36m(keypoints.data) # shape (num_humans, 17, 3), H36M format
show_pose2d = draw_2d_pose_from_image(image, keypoints, os.path.join("./output", "2d_pose_"+image_filename))

image_tensor = pil_to_tensor(Image.open(image)).permute(1, 2, 0).to(device) # shape (Hight, Width, Channels)
H, W = image_tensor.shape[0], image_tensor.shape[1]

pose3d_model = DSTformer(dim_in=3, dim_out=3, dim_feat=512, dim_rep=512,
                         depth=5, num_heads=8, mlp_ratio=2, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                         maxlen=243, num_joints=17).to(device)
pose3d_model = nn.DataParallel(pose3d_model)

config = "models/pt/MB_ft_h36m.yaml"
checkpoint = "models/pt/MB_best_epoch.bin"
args = get_config(config)
checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
pose3d_model.load_state_dict(checkpoint["model_pos"], strict=True)
pose3d_model.eval()

scale = min(W,H) / 2.0
keypoints[:,:,:2] = keypoints[:,:,:2] - np.array([W, H]) / 2.0
keypoints[:,:,:2] = keypoints[:,:,:2] / scale
keypoints_3D = keypoints.astype(np.float32)[0] # shape (1, 17, 3), single worker 

with torch.no_grad():
    keypoints_3D = torch.from_numpy(np.expand_dims(np.expand_dims(keypoints_3D, axis=0), axis=0)).to(device)
    predicted_3d = pose3d_model(keypoints_3D).squeeze(0).cpu().detach().numpy() # shape (1, 17, 3)
    render_and_save(predicted_3d, save_path=os.path.join("./output", "3d_pose_"+image_filename.split(".")[0]+".png"))
    np.save(os.path.join("./output", "3d_pose_"+image_filename.split(".")[0]+".npy"), predicted_3d)
```

### Ergonomic Risk Assessment (REBA/OWAS)
```python
from ergonomics.reba import RebaScore
from ergonomics.owas import OwasScore
from data.utils import h36m2rcoco

# predicted_3d = np.load("../output/3D_pose_worker.npy")
pose = np.squeeze(h36m2rcoco(predicted_3d))
rebaScore = RebaScore()
REBA_body_params = rebaScore.get_body_angles_from_pose_right(pose)
REBA_arms_params = rebaScore.get_arms_angles_from_pose_right(pose)
rebaScore.set_body(REBA_body_params)
score_a, partial_a = rebaScore.compute_score_a()

rebaScore.set_arms(REBA_arms_params)
score_b, partial_b = rebaScore.compute_score_b()

score_c, caption = rebaScore.compute_score_c(score_a, score_b)

print("Score A: ", score_a, "Partial: ", partial_a)
print("Score B: ", score_b, "Partial: ", partial_b)
print("Score C: ", score_c, caption)

owasScore = OwasScore()
OWAS_body_params = owasScore.get_param_from_pose(pose, verbose=False)
owasScore.set_body_params(OWAS_body_params)
owas_score, partial_score = owasScore.compute_score()

print("Owas Score:", owas_score)
print("Trunk, Arms, Legs, Load :", partial_score)
```

## License
- Academic purpose only.

## Acknowledgements

- [Gunwoo Yong](https://github.com/gwyong)
- [Leyang Wen](https://github.com/LeyangWen)
- [Francis Baek](https://www.linkedin.com/in/francis-baek-58789b233/)

## Base Repositories

Our codes are based on the below repositories.
- [YOLOv8](https://github.com/ultralytics/ultralytics)
- [MotionBERT](https://github.com/Walter0807/MotionBERT)
- [REBA_OWAS](https://github.com/rs9000/ergonomics)