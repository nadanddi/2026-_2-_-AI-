from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).with_name('scan_debug_v1.py')),run_name='__main__',init_globals={'STAGE':'probe','LIMIT':2})
