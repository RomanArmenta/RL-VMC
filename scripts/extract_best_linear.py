import pandas as pd
import sys
try:
    df = pd.read_csv(sys.argv[1])
    best = df.sort_values("G_RL", ascending=False).iloc[0]
    print(f"{best['actor_amplitude']},{best['penalty_factor']},{best['sigma_scale']},{best['max_phys_jump']}")
except Exception as e:
    sys.exit(1)
