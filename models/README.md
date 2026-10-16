# Models

Trained checkpoints go here. `.pth` files are git-ignored because of their size.

`notebooks/03_train_model.ipynb` saves the checkpoint with the best validation mIoU as `deeplabv3_resnet50_coral.pth`, and `notebooks/04_inference_ui.ipynb` loads it.

## Checkpoint format

```python
{"model": state_dict, "num_classes": 40}
```

Loading it:

```python
import torch
from torchvision.models.segmentation import deeplabv3_resnet50

ckpt = torch.load("models/deeplabv3_resnet50_coral.pth", map_location="cpu")
model = deeplabv3_resnet50(num_classes=ckpt["num_classes"])
model.load_state_dict(ckpt["model"])
model.eval()
```

> The notebooks currently use hardcoded absolute paths for `save_path`. Point them here (`../models/deeplabv3_resnet50_coral.pth` from `notebooks/`).
