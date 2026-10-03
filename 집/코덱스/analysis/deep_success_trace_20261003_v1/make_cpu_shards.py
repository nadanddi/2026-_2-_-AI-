from pathlib import Path
H=Path(__file__).resolve().parent
source=(H/'pfn_ec_cpu.py').read_text(encoding='utf-8')
source=source.replace('def main():','def main(folds=(3,6)):').replace('for fold in [3,6]:','for fold in folds:').replace("if __name__=='__main__':main()","if __name__=='__main__':main((int(sys.argv[1]),))")
(H/'pfn_ec_cpu_shard.py').write_text(source,encoding='utf-8')
collector=(H/'collect_pfn.py').read_text(encoding='utf-8').replace("paths=list(root.glob('PFN_*_*.npz'))","paths=list(root.glob('PFN_TEMP_*.npz'))+list((O/'pfn_ec_cpu').glob('PFN_EC_*.npz'))")
(H/'collect_pfn_v2.py').write_text(collector,encoding='utf-8')
