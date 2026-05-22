import re

with open('/mnt/home/vyhong/projects/nova3r/nova3r/models/nova3r_img_cond.py', 'r') as f:
    content = f.read()

import_statement = "from training.training_utils import visualize_extrinsics\n"
if "from training.training_utils import visualize_extrinsics" not in content:
    content = import_statement + content

replacement = """        if not hasattr(self, '_extrinsics_saved') and batch is not None and batch.get("extrinsics") is not None:
            exts = batch["extrinsics"][0].detach().cpu().numpy()
            visualize_extrinsics(exts)
            self._extrinsics_saved = True"""

if "import matplotlib" in content:
    content = re.sub(r'        if not hasattr\(self, \'_extrinsics_saved\'\) and batch is not None and batch\.get\("extrinsics"\) is not None:.*?self\._extrinsics_saved = True', replacement, content, flags=re.DOTALL)

with open('/mnt/home/vyhong/projects/nova3r/nova3r/models/nova3r_img_cond.py', 'w') as f:
    f.write(content)
