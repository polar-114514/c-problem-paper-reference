# -*- coding: utf-8 -*-
"""
第二问：男胎 BMI 合理分组 + 每组最佳 NIPT 时点（机会约束型 / 带权风险函数型）+ 检测误差灵敏度
复用第一问二次 Logit 混合效应模型的固定效应系数与方差分量。

口径（与第一问一致）：
  - 主样本 = 序号<683 且孕周∈[10,26) 男胎，613 事件/167 孕妇；前瞻性用"首测 BMI"
  - 固定效应（logit 尺度，已中心化）：t=孕周-18, B̄-32, d=B-B̄(=0), A=年龄-29, P=生产次数
  - 边际概率：Pr(Y>=0.04|x)=1-Φ((logit(0.04)-η(t))/sqrt(z^TΣz+σ²))

建模链：个体达标周 T_i（边际概率=0.5 处）→ 按 BMI 有序分割分组 → 组达标比例分布 → 每组最优时点。
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from pathlib import Path

# ---------- 第一问模型参数 ----------
B0, B1, B2 = -2.8954, 0.0460, 0.005082
BB, BW = -0.03303, -0.01284
BA, BP = -0.01675, 0.18863
S0, S1, RHO, SIG_RES = 0.3527, 0.03357, 0.0784, 0.2823
LOGIT4 = np.log(0.04 / 0.96)
WEEK_LO, WEEK_HI = 10.0, 25.5
P0 = 0.5          # 个体"达标"判定的边际概率阈值（50%：更可能已达标）
LAM = (1.0, 1.0)  # 风险函数权重 (λ1: 未达标/需重测, λ2: 晚发现)

def total_var(t_c):
    return S0**2 + 2*RHO*S0*S1*t_c + S1**2*t_c**2 + SIG_RES**2

def eta_fixed(week, bmi, age, parity):
    t_c = week - 18.0
    return B0 + B1*t_c + B2*t_c**2 + BB*(bmi-32.0) + BW*0.0 + BA*(age-29.0) + BP*parity

def p_marg(week, bmi, age, parity):
    t_c = week - 18.0
    z = (LOGIT4 - eta_fixed(week, bmi, age, parity)) / np.sqrt(total_var(t_c))
    return 1.0 - norm.cdf(z)

def reach_time(bmi, age, parity):
    """个体达标周：边际概率首次 >=P0 的最小孕周（夹在窗口内）"""
    ws = np.linspace(WEEK_LO, WEEK_HI, 1600)
    p = np.array([p_marg(w, bmi, age, parity) for w in ws])
    idx = np.where(p >= P0)[0]
    return ws[idx[0]] if len(idx) else WEEK_HI

def late_risk(t):
    t = np.asarray(t, dtype=float)
    return np.clip(np.where(t <= 12, 0.0, 0.5 + 0.5*(t-13)/(25-13)), 0.0, 1.0)

def fisher_partition(values, k, min_size=3):
    n = len(values)
    s = np.concatenate([[0.0], np.cumsum(values)]); s2 = np.concatenate([[0.0], np.cumsum(values**2)])
    def sse(i, j):
        c = j - i
        return 0.0 if c <= 0 else (s2[j]-s2[i]) - (s[j]-s[i])**2/c
    dp = np.full((k+1, n+1), np.inf); par = np.zeros((k+1, n+1), dtype=int)
    dp[0, 0] = 0.0
    for kk in range(1, k+1):
        for j in range(kk*3, n+1):
            bi, bv = -1, np.inf
            for i in range((kk-1)*3, j-2):
                v = dp[kk-1, i] + sse(i, j)
                if v < bv: bv, bi = v, i
            dp[kk, j] = bv; par[kk, j] = bi
    bounds, j = [], n
    for kk in range(k, 0, -1):
        i = par[kk, j]; bounds.append((i, j)); j = i
    return list(reversed(bounds))

def choose_k(t_sorted, n_hi=6, min_size=12):
    best, bk = np.inf, None
    for k in range(2, n_hi+1):
        b = fisher_partition(t_sorted, k)
        sz = [x-y for x, y in b]
        if min(sz) < min_size: continue
        sse_tot = sum(((t_sorted[a:bnd]-t_sorted[a:bnd].mean())**2).sum() for a, bnd in b)
        bic = len(t_sorted)*np.log(sse_tot/len(t_sorted)+1e-9) + k*np.log(len(t_sorted))
        if bic < best: best, bk = bic, k
    return bk or n_hi

def solve_group(lo, hi, bmi_arr, age_arr, par_arr):
    """概率型达标比例 p_g(t)=mean_i P(Y>=4%|x_i,t)；就 t 求机会约束型与带权风险函数型最优时点"""
    ws = np.linspace(WEEK_LO, WEEK_HI, 2001)
    ps = np.array([np.mean([p_marg(w, bmi_arr[i], age_arr[i], par_arr[i]) for i in range(lo, hi)]) for w in ws])
    t_cc = {c: float(ws[np.where(ps >= c)[0][0]]) for c in (0.60, 0.70, 0.80, 0.90)}
    R = LAM[0]*(1-ps) + LAM[1]*late_risk(ws)
    t_rk = float(ws[int(np.argmin(R))])
    return t_cc, t_rk, float(R.min())

def main():
    base = Path(r"C:\Users\14820\Documents\deepseek\c-problem-paper-reference")
    df = pd.read_csv(base/"01_第一问"/"02_数据处理"/"关系建模"/"冻结数据"/"第一问主模型冻结样本.csv", encoding="utf-8-sig")
    df = df.sort_values(["孕妇代码", "孕周数"])
    first = df.groupby("孕妇代码", as_index=False).first()
    bmi = first["孕妇体质指数_BMI"].to_numpy(float); age = first["年龄"].to_numpy(float); par = first["生产次数"].to_numpy(float)
    n = len(first)

    T = np.array([reach_time(bmi[i], age[i], par[i]) for i in range(n)])
    order = np.argsort(bmi)
    bs, ts = bmi[order], T[order]

    # 可解释分组：按 BMI 分位数切成 N_BIN 组（单调、样本量均衡、可解释）
    N_BIN = 5
    bins = pd.qcut(bs, N_BIN, duplicates="drop")          # Categorical
    bin_labels = bins.codes.astype(int)                     # 0..nbin-1
    edges = bins.categories
    print(f"前瞻性样本 {n} 人 | BMI [%.1f, %.1f]" % (bs.min(), bs.max()))
    print(f"BMI 分位边界（%d组）：" % len(edges) + ", ".join(f"{int(e.left)}~{int(e.right)}" for e in edges))

    rows = []
    for gi in range(N_BIN):
        a = np.where(bin_labels == gi)[0]
        if len(a) == 0:
            continue
        tcc, trk, rmin = solve_group(int(a.min()), int(a.max())+1, bs, age[order], par[order])
        rows.append(dict(组=gi+1, BMI_range=f"[{bs[a.min()]:.1f},{bs[a.max()]:.1f}]", 样本量=len(a),
                         时点_达标60=round(tcc[0.60],2), 时点_达标70=round(tcc[0.70],2),
                         时点_达标80=round(tcc[0.80],2), 时点_达标90=round(tcc[0.90],2),
                         时点_风险函数=round(trk,2), 最小风险=round(rmin,4)))
    tab = pd.DataFrame(rows)
    print("\n=== 概率型达标比例：每组最优时点（c=60/70/80/90% & 带权风险函数型）===")
    print(tab.to_string(index=False))

    print("\n=== 检测误差灵敏度（达标比例 80%/90% 的最优时点随误差系数）===")
    global SIG_RES
    sens = []
    for fac in [1.0, 1.05, 1.10, 1.20, 1.30]:
        SIG_RES = 0.2823*np.sqrt(fac)
        row = {"误差系数": fac}
        for gi in range(N_BIN):
            a = np.where(bin_labels == gi)[0]
            if len(a) == 0:
                continue
            tcc, _, _ = solve_group(int(a.min()), int(a.max())+1, bs, age[order], par[order])
            row[f"组{gi+1}_c80"] = round(tcc[0.80], 2)
        sens.append(row)
    print(pd.DataFrame(sens).to_string(index=False))

    out = Path(r"C:\Users\14820\Documents\deepseek\数学建模\NIPT\问题二_时点优化")
    out.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out/"分组与时点结果.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(sens).to_csv(out/"误差灵敏度.csv", index=False, encoding="utf-8-sig")
    print(f"\n已输出到 {out}")

if __name__ == "__main__":
    main()
