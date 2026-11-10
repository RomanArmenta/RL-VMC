#!/usr/bin/env python3
"""
Compiles NODAL_CROSSING_BENCHMARKS_REPORT.md into a publication-quality LaTeX document
and builds the final PDF using XeLaTeX.
"""

import os
import re
import subprocess

MD_FILE = "NODAL_CROSSING_BENCHMARKS_REPORT.md"
TEX_FILE = "NODAL_CROSSING_BENCHMARKS_REPORT.tex"
PDF_FILE = "NODAL_CROSSING_BENCHMARKS_REPORT.pdf"
XELATEX = "/usr/local/texlive/2024/bin/universal-darwin/xelatex"

def escape_code_span(text):
    mapping = {
        '\\': r'\textbackslash{}',
        '{': r'\{',
        '}': r'\}',
        '$': r'\$',
        '&': r'\&',
        '%': r'\%',
        '#': r'\#',
        '_': r'\_',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
        '<': r'\textless{}',
        '>': r'\textgreater{}',
    }
    return "".join(mapping.get(ch, ch) for ch in text)

def process_prose_line(line):
    # 1. Protect inline code `...`
    codes = []
    def save_code(m):
        codes.append(m.group(1))
        return f"ZZZCODE{len(codes)-1}ZZZ"
    line = re.sub(r'`([^`]+)`', save_code, line)

    # 2. Protect inline math $...$
    maths = []
    def save_math(m):
        maths.append(m.group(0))
        return f"ZZZMATH{len(maths)-1}ZZZ"
    line = re.sub(r'(\$[^\$]+\$)', save_math, line)

    # 3. Escape LaTeX special characters in remaining prose
    line = line.replace('&', r'\&')
    line = line.replace('%', r'\%')
    line = line.replace('#', r'\#')
    line = line.replace('_', r'\_')
    line = line.replace('—', '---')
    line = line.replace('~', r'\textasciitilde{}')

    # Markdown formatting
    line = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', line)
    line = re.sub(r'\*(.+?)\*', r'\\textit{\1}', line)

    # 4. Restore inline code
    for idx, c in enumerate(codes):
        c_esc = escape_code_span(c)
        line = line.replace(f"ZZZCODE{idx}ZZZ", f"\\texttt{{{c_esc}}}")

    # 5. Restore inline math
    for idx, m in enumerate(maths):
        line = line.replace(f"ZZZMATH{idx}ZZZ", m)

    return line

def convert_md_table(table_lines):
    rows = []
    for l in table_lines:
        l = l.strip()
        if l.startswith('|'):
            l = l[1:]
        if l.endswith('|'):
            l = l[:-1]
        cols = [c.strip() for c in l.split('|')]
        rows.append(cols)
    
    if len(rows) < 2:
        return ""
    
    header = rows[0]
    is_sep = all(re.match(r'^:?-+:?$', c) for c in rows[1])
    data_rows = rows[2:] if is_sep else rows[1:]
    
    num_cols = len(header)
    col_spec = "l" + "c" * (num_cols - 1)
    
    tex = []
    tex.append("\\begin{table}[H]")
    tex.append("\\centering")
    tex.append("\\begin{adjustbox}{max width=\\textwidth}")
    tex.append(f"\\begin{{tabular}}{{{col_spec}}}")
    tex.append("\\toprule")
    
    header_str = " & ".join([f"\\textbf{{{process_prose_line(c)}}}" for c in header]) + " \\\\"
    tex.append(header_str)
    tex.append("\\midrule")
    
    for r in data_rows:
        if len(r) < num_cols:
            r = r + [""] * (num_cols - len(r))
        elif len(r) > num_cols:
            r = r[:num_cols]
        r_str = " & ".join([process_prose_line(c) for c in r]) + " \\\\"
        tex.append(r_str)
        
    tex.append("\\bottomrule")
    tex.append("\\end{tabular}")
    tex.append("\\end{adjustbox}")
    tex.append("\\end{table}")
    return "\n".join(tex)

def main():
    with open(MD_FILE, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')
    
    out_lines = []
    
    preamble = r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=1.8cm]{geometry}
\usepackage{amsmath,amssymb,amsfonts,bm}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{adjustbox}
\usepackage{float}
\usepackage{caption}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{microtype}
\usepackage{listings}
\usepackage{tikz}
\usetikzlibrary{shapes,arrows.meta,positioning}

\definecolor{darkblue}{rgb}{0.0,0.15,0.55}
\definecolor{codegray}{rgb}{0.96,0.96,0.96}
\definecolor{codeborder}{rgb}{0.8,0.8,0.8}

\hypersetup{
    colorlinks=true,
    linkcolor=darkblue,
    citecolor=darkblue,
    urlcolor=darkblue,
    pdfauthor={RL-VMC Framework},
    pdftitle={Benchmarks de Cruce Nodal y Muestreo Puro Multi-Determinante}
}

\lstset{
    backgroundcolor=\color{codegray},
    basicstyle=\ttfamily\small,
    breaklines=true,
    frame=single,
    rulecolor=\color{codeborder},
    xleftmargin=0.5cm,
    framexleftmargin=0.3cm
}

\title{\textbf{\Large Benchmarks de Cruce Nodal y Muestreo Puro Multi-Determinante (CASSCF 1D):\\
Teoría, Implementación y Resultados Consolidados}}
\author{\textbf{RL-VMC Framework / Deep Deterministic Quantum Monte Carlo}}
\date{Noviembre 2026}

\begin{document}
\maketitle

\begin{abstract}
Este informe presenta el marco teórico formal, la formulación algorítmica y el análisis cuantitativo consolidado de 114 configuraciones evaluadas en los dos benchmarks fundamentales de muestreo MCMC puro sobre funciones de onda multiconfiguracionales congeladas (CASSCF): el cristal de Wigner fuertemente correlacionado ($N=4, g=4.0, a=0.1$) y el sistema multi-determinante $\mathrm{CAS}(6e, 6o)$ con 20 determinantes en base Hermite expandida ($N=6, g=1.0, N_{\mathrm{basis}}=256$). Se demuestra que MALA clásico colapsa debido al congelamiento topológico y a las seudo-barreras de Wigner ($D_{\mathrm{KL}}^{\mathrm{spin}} \sim 0.02 - 0.63$), mientras que el muestreador adaptativo RL-MALA con red DeepSet y recompensa CDLB erradica el bloqueo de espín ($D_{\mathrm{KL}}^{\mathrm{spin}} \sim 10^{-5} - 10^{-3}$, mejora de hasta $18,860\times$), preservando rigurosamente la energía variacional CASSCF ($\Delta E < 0.5\text{ mHa}$) y cuadruplicando la distancia de salto ($\mathrm{ESJD} > 2.4\text{ Bohr}^2$). Asimismo, se presenta la caracterización sistemática de intercambios topológicos (\texttt{flips}) en función del acoplamiento $g \in [0.1, 8.0]$ y del número de electrones $N$, demostrando aceleraciones de $50\times$ a $160\times$ en la tasa de cruces pozo-a-pozo mediante saltos colectivos de 1 paso ($25\%-55\%$), se justifica físicamente por qué las corridas preliminares con orbitales de core congelados ($n_{\mathrm{core}} > 0$) en $N \ge 6$ impidieron la cristalización de Wigner, y se documenta el protocolo de re-ejecución con estados CASSCF convergidos de espacio activo completo ($N_{\mathrm{basis}}=256, n_{\mathrm{core}}=0$).
\end{abstract}

\tableofcontents
\vspace{0.8cm}
\hrule
\vspace{0.8cm}
"""
    out_lines.append(preamble)
    
    i = 0
    in_code_block = False
    code_lang = ""
    code_lines = []
    
    in_math_block = False
    math_lines = []
    
    in_table = False
    table_lines = []
    
    in_itemize = False
    
    while i < len(lines):
        line = lines[i]
        
        # Check code block
        if line.strip().startswith('```'):
            if in_code_block:
                in_code_block = False
                if code_lang == 'mermaid':
                    tikz_code = r"""
\begin{figure}[H]
\centering
\begin{tikzpicture}[
    node distance=1.2cm,
    block/.style={rectangle, draw=darkblue, fill=blue!8, text centered, rounded corners, minimum height=1cm, minimum width=3.4cm, font=\small\bfseries},
    arrow/.style={thick, ->, >=stealth, darkblue}
]
\node[block] (A) {Actor Entrenado ($g=1.0$)};
\node[block, right=1.2cm of A] (B) {Fase 1: Zero-Shot Transfer ($g=2.0$)};
\node[block, right=1.2cm of B] (C) {Fase 2: Few-Shot Fine-Tuning};
\node[block, below=1.2cm of B] (D) {Fase 3: Aceleración VMC Variacional};

\draw[arrow] (A) -- (B);
\draw[arrow] (B) -- (C);
\draw[arrow] (B) -- (D);
\draw[arrow] (C) |- (D);
\end{tikzpicture}
\caption{Protocolo experimental propuesto para Transfer Learning y aceleración de VMC.}
\end{figure}
"""
                    out_lines.append(tikz_code)
                else:
                    out_lines.append("\\begin{lstlisting}")
                    out_lines.append("\n".join(code_lines))
                    out_lines.append("\\end{lstlisting}\n")
                code_lines = []
                code_lang = ""
                i += 1
                continue
            else:
                in_code_block = True
                code_lang = line.strip()[3:].strip()
                code_lines = []
                i += 1
                continue
                
        if in_code_block:
            code_lines.append(line)
            i += 1
            continue

        # Check math block $$
        if line.strip().startswith('$$'):
            if in_math_block:
                in_math_block = False
                out_lines.append("\\[")
                out_lines.append("\n".join(math_lines))
                out_lines.append("\\]\n")
                math_lines = []
                i += 1
                continue
            else:
                stripped = line.strip()
                if stripped.endswith('$$') and len(stripped) > 4:
                    expr = stripped[2:-2].strip()
                    out_lines.append(f"\\[\n{expr}\n\\]\n")
                    i += 1
                    continue
                else:
                    in_math_block = True
                    math_lines = []
                    i += 1
                    continue
                    
        if in_math_block:
            math_lines.append(line)
            i += 1
            continue
            
        # Check table
        if line.strip().startswith('|'):
            if in_itemize:
                out_lines.append("\\end{itemize}\n")
                in_itemize = False
            if not in_table:
                in_table = True
                table_lines = [line]
            else:
                table_lines.append(line)
            i += 1
            continue
        else:
            if in_table:
                in_table = False
                out_lines.append(convert_md_table(table_lines))
                table_lines = []
                
        # Check image ![caption](path)
        img_match = re.match(r'^!\[(.*?)\]\((.*?)\)', line.strip())
        if img_match:
            if in_itemize:
                out_lines.append("\\end{itemize}\n")
                in_itemize = False
            caption = img_match.group(1)
            img_path = img_match.group(2)
            ext_caption = ""
            if i + 1 < len(lines) and lines[i+1].strip().startswith('*Figura'):
                ext_caption = lines[i+1].strip().strip('*')
                i += 1
            
            final_cap = ext_caption if ext_caption else caption
            final_cap_tex = process_prose_line(final_cap)
            
            tex_fig = f"""
\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.88\\textwidth]{{{img_path}}}
\\caption{{{final_cap_tex}}}
\\end{{figure}}
"""
            out_lines.append(tex_fig)
            i += 1
            continue
            
        # Check section headers
        if line.startswith('# '):
            i += 1
            continue
        elif line.startswith('## '):
            if in_itemize:
                out_lines.append("\\end{itemize}\n")
                in_itemize = False
            title = line[3:].strip()
            title_clean = re.sub(r'^\d+\.\s*', '', title)
            out_lines.append(f"\n\\section{{{process_prose_line(title_clean)}}}\n")
            i += 1
            continue
        elif line.startswith('### '):
            if in_itemize:
                out_lines.append("\\end{itemize}\n")
                in_itemize = False
            title = line[4:].strip()
            title_clean = re.sub(r'^\d+\.\d+\s*', '', title)
            out_lines.append(f"\n\\subsection{{{process_prose_line(title_clean)}}}\n")
            i += 1
            continue
        elif line.startswith('#### '):
            if in_itemize:
                out_lines.append("\\end{itemize}\n")
                in_itemize = False
            title = line[5:].strip()
            out_lines.append(f"\n\\subsubsection{{{process_prose_line(title)}}}\n")
            i += 1
            continue
        elif line.startswith('##### '):
            if in_itemize:
                out_lines.append("\\end{itemize}\n")
                in_itemize = False
            title = line[6:].strip()
            out_lines.append(f"\n\\paragraph{{{process_prose_line(title)}}}\n")
            i += 1
            continue
            
        # Horizontal rules
        if line.strip() in ['---', '***', '___']:
            if in_itemize:
                out_lines.append("\\end{itemize}\n")
                in_itemize = False
            out_lines.append("\n\\vspace{0.4cm}\\hrule\\vspace{0.4cm}\n")
            i += 1
            continue
            
        # Lists and paragraphs
        stripped = line.strip()
        if stripped:
            num_match = re.match(r'^\d+\.\s+(.*)', stripped)
            bullet_match = re.match(r'^[-*]\s+(.*)', stripped)
            
            if num_match or bullet_match:
                if not in_itemize:
                    out_lines.append("\\begin{itemize}")
                    in_itemize = True
                content_item = num_match.group(1) if num_match else bullet_match.group(1)
                out_lines.append(f"\\item {process_prose_line(content_item)}")
            else:
                if in_itemize:
                    out_lines.append("\\end{itemize}\n")
                    in_itemize = False
                out_lines.append(process_prose_line(line) + "\n")
        else:
            if in_itemize:
                out_lines.append("\\end{itemize}\n")
                in_itemize = False
            out_lines.append("\n")
            
        i += 1

    if in_itemize:
        out_lines.append("\\end{itemize}\n")
        
    out_lines.append("\n\\end{document}\n")
    
    tex_content = "\n".join(out_lines)
    
    with open(TEX_FILE, 'w', encoding='utf-8') as f:
        f.write(tex_content)
        
    print(f"Generated {TEX_FILE} successfully.")
    
    print(f"Compiling with {XELATEX} (Pass 1)...")
    res1 = subprocess.run([XELATEX, "-halt-on-error", TEX_FILE], capture_output=True, text=True)
    if res1.returncode != 0:
        print("XeLaTeX Pass 1 FAILED!")
        # print error context
        err_lines = [l for l in res1.stdout.split('\n') if l.startswith('!') or 'Error' in l or 'l.' in l]
        print("Error summary:\n", "\n".join(err_lines[:15]))
        print("\nTail of output:\n", res1.stdout[-1500:])
        return False
        
    print(f"Compiling with {XELATEX} (Pass 2 for TOC/Cross-References)...")
    res2 = subprocess.run([XELATEX, "-halt-on-error", TEX_FILE], capture_output=True, text=True)
    if res2.returncode != 0:
        print("XeLaTeX Pass 2 FAILED!")
        return False
        
    print(f"Successfully generated {PDF_FILE}!")
    return True

if __name__ == '__main__':
    main()
