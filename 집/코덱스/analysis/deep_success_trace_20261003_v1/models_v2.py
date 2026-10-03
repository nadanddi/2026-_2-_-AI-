from models import *
from models import ECModel as OriginalECModel
class ECModel(OriginalECModel):
 def __getstate__(self):
  state=self.__dict__.copy();state.pop('core',None);return state
