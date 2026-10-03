import shutil
import run_models
from models_v3 import ECModel
if __name__=='__main__':
 old=run_models.O;run_models.O=old/'ec_v3';run_models.O.mkdir(exist_ok=True)
 if not (run_models.O/'world.joblib').exists():shutil.copy2(old/'world.joblib',run_models.O/'world.joblib')
 run_models.H=run_models.H/'ec_v3';run_models.H.mkdir(exist_ok=True)
 run_models.ECModel=ECModel;run_models.main('EC')
