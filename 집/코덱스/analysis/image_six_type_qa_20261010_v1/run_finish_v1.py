import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import train_contract_v3
train_contract_v3.fit('finish')
