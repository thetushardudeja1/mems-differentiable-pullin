"""
Emit IEEE-format LaTeX tables (T1-T3) from the measured results.

Every number here was produced by a script in this directory; the source is
named in the comment beside each row so any value can be traced and re-run.
Written to ../figures/tables.tex for direct \\input{} into the report.
"""

import os

OUT = "../figures"
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- T1
# sources: test_fold.py, validate_beam.py, model2dof.py,
#          validate_nazemi.py, validate_mtest.py
T1 = r"""
\begin{table}[t]
\caption{Solver validation against eight independent sources, 1967--2025.
``mech.'' is the predicted instability mechanism; the last two rows are
\emph{measured} devices. $^\dagger$Normalised, as in \cite{nazemi}.}
\label{tab:validation}
\centering\scriptsize
\setlength{\tabcolsep}{2pt}
\begin{tabular}{@{}llrrr@{}}
\toprule
Source & Quantity & Ours & Ref. & Error \\
\midrule
Closed form           & $\lambda^*=4/27$ (lumped) & 0.148150 & 0.148148 & 0.001\% \\
Nathanson \cite{nathanson} & $\Lambda_{\rm PI}$, cantilever & 1.6799 & 1.6790 & 0.05\%  \\
Nathanson \cite{nathanson} & $\Lambda_{\rm PI}$, fix--fix   & 69.935 & 71.167 & 1.73\%  \\
Analytic $c^3$ law    & $\sum \partial\Lambda/\partial D$ & 5.0398 & 5.0398 & \textbf{0.00\%} \\
Seeger \cite{seeger}  & tip-in, design \#1        & 19.0\%   & 19\%     & --- \\
Seeger \cite{seeger}  & tip-in, design \#3        & 85.1\%   & 85\%     & --- \\
Seeger \cite{seeger}  & mech., 3 designs          & \textbf{3/3} & 3/3 & --- \\
Nazemi \cite{nazemi}  & $V_{\rm PI}$ reduction$^\dagger$ & 13.2\% & 16.0\% & 2.8\,pp \\
M-TEST \cite{osterberg} & $V_{\rm PI}$, 500\,\textmu m & 11.70\,V & 11.1$\pm$0.1\,V & +5.4\% \\
\bottomrule
\end{tabular}
\end{table}
"""

# ---------------------------------------------------------------- T2
# source: inverse_design.py (both boundary conditions, N=60)
T2 = r"""
\begin{table}[t]
\caption{Inverse design versus \cite{haluzan}, both boundary conditions, every
design re-fitted to 2\,\textmu m stable travel at $N=60$.}
\label{tab:inverse}
\centering\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lrrrr@{}}
\toprule
& \multicolumn{2}{c}{Cantilever} & \multicolumn{2}{c}{Fixed--fixed} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Design & Ours & Paper & Ours & Paper \\
\midrule
Uniform gap            & 22.86 & 23.66 & 175.83 & 178.26 \\
Linear ($n{=}1$)       & 14.31 & 14.86 & 123.01 & 125.48 \\
Best family in \cite{haluzan} & 13.61 & 14.14 & 121.79 & 123.94 \\
Free-form (ours)       & \textbf{13.60} & --- & \textbf{120.03} & --- \\
\midrule
Optimal exponent $n^*$ & 1.29 & 4/3 & 1.00 & 1.00 \\
Gain over best family  & 0.14\% & --- & 1.45\% & --- \\
\bottomrule
\end{tabular}
\end{table}
"""

# ---------------------------------------------------------------- T3
# sources: surrogate_fair.py, fno_surrogate.py, amortized_hard.py,
#          amortized_vs_converged.py
# T3 and T4 are deliberately TWO single-column tables and not one table*.
# As a single wide table this occupied the full page width, which a 3-page
# limit cannot afford; split, each half fits one column and text flows past.
T3 = r"""
\begin{table}[t]
\caption{Design methodologies compared. \cite{zhang} values are as reported
there, for a different task, so the comparison is of methodology. Rows 3--8
share one task at an equal 400-solve budget, scored by the exact solver.}
\label{tab:methods}
\centering\scriptsize
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{@{}llrcrr@{}}
\toprule
 & Method & Sims.\ in & Gradient & Time per & $V_{\rm PI}$ \\
 & & dataset & source & design & [V] \\
\midrule
\cite{zhang} & MLP fitted to FEM data & 48{,}000 & surrogate & 0.01\,s & --- \\
\cite{haluzan} & Hand-tuned gap family & --- & none & manual & 14.14 \\
\midrule
 & MLP surrogate $+$ act.\ learn. & 400 & surrogate & 71\,s & 17.20 \\
 & FNO surrogate $+$ act.\ learn. & 400 & surrogate & 594\,s & 14.25 \\
\midrule
\textbf{This} & Direct differentiable & \textbf{0} & \textbf{exact} & 15\,s
  & 13.78 \\
\textbf{work} & Direct, 4 restarts & \textbf{0} & \textbf{exact} & 170\,s
  & 13.67 \\
 & Amortized network & \textbf{0} & \textbf{exact} & \textbf{0.04\,ms}
  & 13.64 \\
 & Network $+$ 100 steps & \textbf{0} & \textbf{exact} & 17\,s
  & \textbf{13.60} \\
\bottomrule
\end{tabular}
\end{table}
"""

# ---------------------------------------------------------------- T4
# sources: surrogate_fair.py, fno_surrogate.py, amortized_hard.py,
#          amortized_vs_converged.py
T4 = r"""
\begin{table}[t]
\caption{All methods on one task (cantilever, 2\,\textmu m travel) at an equal
budget of 400 true solves, scored by the exact solver. Surrogates were given
active learning, a trust region and feasible restarts. Times exclude the
network's one-time training (1577\,s, 6000 live solves).}
\label{tab:budget}
\centering\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lrrr@{}}
\toprule
Method & Samples & Time & $V_{\rm PI}$ [V] \\
\midrule
MLP surrogate $+$ act.\ learning & 400 & 71\,s & 17.20 \\
FNO surrogate $+$ act.\ learning & 400 & 594\,s & 14.25 \\
Direct differentiable, 400 solves & \textbf{0} & 15\,s & 13.78 \\
Direct, 4 restarts & \textbf{0} & 170\,s & 13.67 \\
Amortized network & \textbf{0} & \textbf{0.04\,ms} & 13.64 \\
Network $+$ 100 steps & \textbf{0} & 17\,s & \textbf{13.60} \\
\midrule
\multicolumn{4}{@{}l@{}}{\scriptsize Hand-tuned design of \cite{haluzan}:
14.14\,V.} \\
\bottomrule
\end{tabular}
\end{table}
"""

if __name__ == "__main__":
    with open(f"{OUT}/tables.tex", "w") as f:
        f.write("% Auto-generated by sim/make_tables.py -- do not edit by hand.\n")
        f.write("% Requires: \\usepackage{booktabs}\n")
        # T4 is not emitted into the report: T3 now carries the equal-budget
        # rows alongside the published comparison, so T4 would repeat them.
        # It is still written out for the Results & Validation document, which
        # has no page limit.
        for t in (T1, T2, T3):
            f.write(t)
            f.write("\n")
    with open(f"{OUT}/tables_extra.tex", "w") as f:
        f.write(T4)
    print(f"wrote {OUT}/tables.tex (T1, T2, T3) and tables_extra.tex (T4)")
    for name, t in [("T1 validation", T1), ("T2 inverse design", T2),
                    ("T3 methodologies", T3), ("T4 equal budget", T4)]:
        rows = [l for l in t.splitlines()
                if "&" in l and "multicolumn" not in l and "cmidrule" not in l]
        print(f"  {name}: {len(rows)} rows")
