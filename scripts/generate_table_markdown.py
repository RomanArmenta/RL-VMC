import pandas as pd

df = pd.read_csv('results_nodal_crossing_search_ayler/csv_summary/nodal_crossing_summary_MultiDet_N6_g1.0.csv')
for col in ['tau', 'actor_amplitude', 'penalty_factor', 'sigma_action_scale', 'acceptance', 'ESJD', 'tau_int_E', 'ESS_per_sec', 'kl_casscf_total_mean', 'kl_casscf_spin_mean', 'E_mean', 'SE_E', 'delta_E_vs_CASSCF']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df_mala = df[~df['is_neural'].astype(str).str.lower().isin(['true', '1'])].sort_values('tau')
df_rl = df[df['is_neural'].astype(str).str.lower().isin(['true', '1'])].sort_values('kl_casscf_spin_mean')

lines = []
lines.append('| Modelo | $\\tau$ | $A_\\mu$ | $\\lambda_{\\mathrm{pen}}$ | $\\sigma_{\\mathrm{sc}}$ | $\\langle E \\rangle \\pm \\mathrm{SE}(E)$ (Ha) | $\\Delta E$ (mHa) | Aceptación | ESJD ($\\text{Bohr}^2$) | $\\tau_{\\mathrm{int}}(E)$ | $\\mathrm{ESS/s}$ ($\\text{s}^{-1}$) | $D_{\\mathrm{KL}}^{\\mathrm{tot}}$ | $D_{\\mathrm{KL}}^{\\mathrm{spin}}$ (Cruce Nodal) |')
lines.append('| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |')

for _, r in df_mala.iterrows():
    lines.append(f"| `{r['label']}` | {r['tau']:.2f} | — | — | — | ${r['E_mean']:.6f} \\pm {r['SE_E']:.6f}$ | ${r['delta_E_vs_CASSCF']*1e3:+.2f}$ | {r['acceptance']*100:.1f}% | {r['ESJD']:.3f} | {r['tau_int_E']:.2f} | {r['ESS_per_sec']:,.0f} | {r['kl_casscf_total_mean']:.2e} | **{r['kl_casscf_spin_mean']:.4f}** |")

for _, r in df_rl.iterrows():
    bold = '**' if r['kl_casscf_spin_mean'] < 5e-5 else ''
    lines.append(f"| {bold}`{r['label']}`{bold} | {r['tau']:.2f} | {r['actor_amplitude']:.2f} | {r['penalty_factor']:.2f} | {r['sigma_action_scale']:.2f} | ${r['E_mean']:.6f} \\pm {r['SE_E']:.6f}$ | ${r['delta_E_vs_CASSCF']*1e3:+.2f}$ | {r['acceptance']*100:.1f}% | {r['ESJD']:.3f} | {r['tau_int_E']:.2f} | {r['ESS_per_sec']:,.0f} | {r['kl_casscf_total_mean']:.2e} | {bold}{r['kl_casscf_spin_mean']:.4e}{bold} |")

output_str = '\n'.join(lines)
with open('figures_param_search_summary/MultiDet_N6_table_ayler.md', 'w') as f:
    f.write(output_str)

print(f"Generated table with {len(lines)-2} models.")
