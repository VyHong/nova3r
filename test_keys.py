import zipfile
import pickle
import sys
import types
from training.data.datasets.replica_utils import igibson_utils
from third_party.triposg.triposg import utils as triposg_utils
sys.modules['utils'] = types.ModuleType('utils')
sys.modules['utils.igibson_utils'] = igibson_utils
with zipfile.ZipFile('datasets/ReplicaPano/frl_apartment_0_000.zip', 'r') as z:
    for name in z.namelist():
        if name.endswith('data.pkl'):
            with z.open(name) as f:
                data = pickle.load(f)
                if 'layout' in data:
                    print(data['layout'].keys())
                    print("Sample for layout keys:")
                    for k, v in data['layout'].items():
                        print(f"{k}: {type(v)}")
                        if isinstance(v, object) and not isinstance(v, (dict, list, int, float, str, type(None))):
                            print(f"{k} has attributes: {dir(v)}")
                else:
                    print("No layout key in data.pkl")
            break
