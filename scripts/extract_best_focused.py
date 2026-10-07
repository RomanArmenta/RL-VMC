import sys, re
try:
    with open(sys.argv[1], 'r') as f: dirname = f.read().strip()
    m = re.search(r'tau([0-9.]+)_amp([0-9.]+)_pen([0-9.]+)_sig([0-9.]+)_jmp([0-9.]+)', dirname)
    print(f"{m.group(1)},{m.group(2)},{m.group(3)},{m.group(4)},{m.group(5)}")
except Exception:
    sys.exit(1)
