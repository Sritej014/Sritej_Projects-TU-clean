#!/usr/bin/env python3
# ============================================================
# BOXED HIGH-k TAIL DIAGNOSTIC ONLY
# ------------------------------------------------------------
# Minimal version:
#   - reads OpenFOAM ASCII fields C, V, subtract(U,UMean)
#   - builds x-lines on unstructured mesh
#   - computes x-only spectra
#   - forms premultiplied spectra
#   - extracts ONLY the boxed tail diagnostic
#   - saves:
#       1) boxed-tail plot PNG
#       2) boxed-tail TXT
#       3) validation TXT
#       4) optional NPZ
#
# IMPORTANT:
#   For the boxed tail plot, x-axis is plotted in LINEAR scale because
#   the inspected range is narrow: kx*delta in [100, 180].
#   This avoids the ugly overflowing tick labels from semilogx.
# ============================================================

import os
import re
import glob
import mmap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ================= USER SETTINGS =================
caseDir = "/WORK/sritej/TASKFILES/Task1/FOR_EXPERIMENT/planar"

timeDirs = [
    "0.81295453", "0.81669453", "0.82043453", "0.82417453", "0.82791453"
]
time_glob = "[0-9]*"
max_times = 50

Nx_expected = 372

uPrimeName = "subtract(U,UMean)"
Cname = "C"
Vname = "V"
useV = True

Lx_user = 1.860000000000e-01
Ly_user = 6.200000000000e-02
Lz_user = 6.200000000000e-02

u_tau = 0.81
nu = 18.4e-06
delta = None

tol_y = None
tol_z = None

dplus_min = 0.5
dplus_max = 3000.0
Nd_bins = 35

smooth_k_plot = 5

outDir = os.getcwd()
save_npz = True

# ----- box diagnostic settings -----
ENABLE_BOX_DIAGNOSTIC = True
BOX_KXD = (100.0, 180.0)
BOX_DPLUS = (8.0, 30.0)
BOX_FIELD = "uu"   # "uu", "vv", "ww", "tot"
TAIL_CHECK_KXD_MIN = 100.0
TAIL_ZERO_THRESHOLD = 0.05


# ================= BASIC HELPERS =================
def moser_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.linewidth": 1.2,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "font.size": 12,
        "mathtext.default": "it",
    })


def moving_average_1d(a, w):
    if w is None or w <= 1:
        return a
    w = int(w)
    if w % 2 == 0:
        w += 1
    k = np.ones(w, dtype=np.float64) / w
    return np.convolve(a, k, mode="same")


def one_sided_factor(Nx):
    Nk = Nx // 2 + 1
    fac = np.ones(Nk, dtype=np.float64)
    if Nk > 2:
        fac[1:-1] = 2.0
    return fac


def _extract_internal_nonuniform_block(mm: mmap.mmap, list_type: bytes):
    idx = mm.find(b"internalField")
    if idx < 0:
        raise RuntimeError("internalField not found")

    pat = b"nonuniform List<" + list_type + b">"
    idx2 = mm.find(pat, idx)
    if idx2 < 0:
        raise RuntimeError(f"nonuniform List<{list_type.decode()}> not found after internalField")

    header = mm[idx2: idx2 + 65536]
    m = re.search(rb"\n\s*(\d+)\s*\n\s*\(\s*\n", header)
    if not m:
        m = re.search(rb"\n\s*(\d+)\s*\n\s*\(\s*", header)
    if not m:
        raise RuntimeError("Could not parse list length and opening bracket")

    N = int(m.group(1))
    data_start = idx2 + m.end()

    end1 = mm.find(b"\n)\n;", data_start)
    end2 = mm.find(b")\n;", data_start)
    if end1 < 0 and end2 < 0:
        raise RuntimeError("Could not find end of list ') ;'")

    data_end = end1 if (end1 >= 0 and (end1 < end2 or end2 < 0)) else end2
    return N, data_start, data_end


def read_foam_vector_field(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            N, a, b = _extract_internal_nonuniform_block(mm, b"vector")
            raw = mm[a:b]
        finally:
            mm.close()

    trans = bytes.maketrans(b"();", b"   ")
    cleaned = raw.translate(trans).decode("ascii", errors="ignore")
    vals = np.fromstring(cleaned, sep=" ", dtype=np.float64, count=3 * N)
    if vals.size != 3 * N:
        raise RuntimeError(f"Vector parse failed in {path}: got {vals.size}, expected {3*N}")
    return vals.reshape(N, 3)


def read_foam_scalar_field(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            N, a, b = _extract_internal_nonuniform_block(mm, b"scalar")
            raw = mm[a:b]
        finally:
            mm.close()

    trans = bytes.maketrans(b"();", b"   ")
    cleaned = raw.translate(trans).decode("ascii", errors="ignore")
    vals = np.fromstring(cleaned, sep=" ", dtype=np.float64, count=N)
    if vals.size != N:
        raise RuntimeError(f"Scalar parse failed in {path}: got {vals.size}, expected {N}")
    return vals


def find_field_path(caseDir, t0, name):
    candidates = [
        os.path.join(caseDir, t0, name),
        os.path.join(caseDir, "0", name),
        os.path.join(caseDir, "constant", name),
        os.path.join(caseDir, "constant", "polyMesh", name),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def infer_tol_from_levels(coord):
    s = np.sort(coord.astype(np.float64))
    ds = np.diff(s)
    ds = ds[ds > 0]
    if ds.size == 0:
        return 1e-9
    dmin = np.percentile(ds, 1.0)
    return 0.25 * dmin


def build_x_planes_and_lines(C, Vw, Nx_expected, tol_y=None, tol_z=None):
    x = C[:, 0].astype(np.float64)
    y = C[:, 1].astype(np.float64)
    z = C[:, 2].astype(np.float64)

    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    zmin, zmax = float(z.min()), float(z.max())

    tol_x = infer_tol_from_levels(x)
    xq = np.round(x / tol_x).astype(np.int64)
    uxq, invx = np.unique(xq, return_inverse=True)
    x_levels = np.array([np.mean(x[xq == k]) for k in uxq], dtype=np.float64)

    sx = np.argsort(x_levels)
    x_levels = x_levels[sx]
    inv_sx = np.empty_like(sx)
    inv_sx[sx] = np.arange(sx.size)
    x_idx = inv_sx[invx]

    Nx = x_levels.size
    if Nx_expected is not None and Nx != Nx_expected:
        raise RuntimeError(
            f"Detected Nx={Nx} x-planes, expected Nx={Nx_expected}. "
            f"If intentional, set Nx_expected=None."
        )

    if tol_y is None:
        tol_y = infer_tol_from_levels(y)
    if tol_z is None:
        tol_z = infer_tol_from_levels(z)

    yq = np.round(y / tol_y).astype(np.int64)
    zq = np.round(z / tol_z).astype(np.int64)
    zq0 = zq - zq.min()
    key = (yq.astype(np.int64) << 32) + zq0.astype(np.int64)

    order = np.argsort(key, kind="mergesort")
    key_sorted = key[order]
    uniq_keys, starts, counts = np.unique(key_sorted, return_index=True, return_counts=True)

    y_rep = np.zeros(uniq_keys.size, dtype=np.float64)
    z_rep = np.zeros(uniq_keys.size, dtype=np.float64)
    w_rep = np.zeros(uniq_keys.size, dtype=np.float64)

    for i in range(uniq_keys.size):
        sl = order[starts[i]:starts[i] + counts[i]]
        w = Vw[sl]
        ws = np.sum(w)
        if ws <= 0:
            w = np.ones(sl.size)
            ws = float(sl.size)

        y_rep[i] = np.sum(w * y[sl]) / ws
        z_rep[i] = np.sum(w * z[sl]) / ws
        w_rep[i] = ws

    meta = {
        "xmin": xmin, "xmax": xmax,
        "ymin": ymin, "ymax": ymax,
        "zmin": zmin, "zmax": zmax,
        "tol_x": tol_x, "tol_y": tol_y, "tol_z": tol_z,
    }

    groups = {
        "order": order,
        "uniq_keys": uniq_keys,
        "starts": starts,
        "counts": counts,
        "y_rep": y_rep,
        "z_rep": z_rep,
        "w_rep": w_rep,
    }

    return Nx, x_levels, x_idx, groups, meta


# ================= BOX DIAGNOSTIC ONLY =================
def boxed_tail_diagnostic(
    kxd,
    d_cent,
    Q,
    bin_nt,
    out_png,
    out_txt,
    field_label=r"k_x\Phi_{uu}/u_\tau^2",
    box_kxd=(100.0, 180.0),
    box_dplus=(8.0, 30.0),
    tail_kmin=100.0,
    zero_threshold=0.05,
):
    """
    Box diagnostic in (kx*delta, d+) space.

    Output metrics:
      tail_mean
      tail_last
      frac_nonincreasing
      R_tail = <Q>_(kx*delta in [100,180]) / max_k <Q>
    """
    K = kxd[1:]
    A = Q[:, 1:]

    mk = (K >= box_kxd[0]) & (K <= box_kxd[1])
    md = (d_cent >= box_dplus[0]) & (d_cent <= box_dplus[1]) & (bin_nt > 0)

    if not np.any(mk):
        print("WARNING: no kx*delta points inside box.")
        return np.nan, np.nan, np.nan, np.nan
    if not np.any(md):
        print("WARNING: no d+ bins inside box.")
        return np.nan, np.nan, np.nan, np.nan

    A_box = A[np.ix_(md, mk)]
    K_box = K[mk]
    D_box = d_cent[md]

    mean_curve = np.nanmean(A_box, axis=0)
    median_curve = np.nanmedian(A_box, axis=0)

    mtail = K_box >= tail_kmin
    K_tail = K_box[mtail]
    tail_curve = mean_curve[mtail]

    tail_last = np.nan
    tail_mean = np.nan
    frac_nonincreasing = np.nan
    r_tail = np.nan

    finite_mean = np.isfinite(mean_curve)
    peak_mean_curve = float(np.nanmax(mean_curve)) if np.any(finite_mean) else np.nan

    finite_tail = np.isfinite(tail_curve)
    if np.any(finite_tail):
        tail_valid = tail_curve[finite_tail]
        tail_last = float(tail_valid[-1])
        tail_mean = float(np.nanmean(tail_valid))

        if tail_valid.size >= 2:
            diffs = np.diff(tail_valid)
            frac_nonincreasing = float(np.sum(diffs <= 0.0) / diffs.size)

        if np.isfinite(peak_mean_curve) and peak_mean_curve > 0.0:
            r_tail = tail_mean / peak_mean_curve

    fig, ax = plt.subplots(figsize=(7.0, 5.4))

    for i, dval in enumerate(D_box):
        row = A_box[i, :]
        if np.any(np.isfinite(row)):
            label = rf"$d^+\approx{dval:.0f}$" if i < 8 else None
            ax.plot(K_box, row, lw=1.0, alpha=0.45, label=label)

    ax.plot(K_box, mean_curve, "k-", lw=2.4, label="box mean")
    ax.plot(K_box, median_curve, "k--", lw=1.7, label="box median")
    ax.axvline(tail_kmin, color="r", linestyle="--", lw=1.2,
               label=rf"tail start: $k_x\delta={tail_kmin:g}$")
    ax.axhline(zero_threshold, color="gray", linestyle=":", lw=1.2,
               label=f"near-zero threshold = {zero_threshold:g}")

    txt = []
    if np.isfinite(tail_mean):
        txt.append(rf"$\langle Q\rangle_{{tail}}={tail_mean:.3e}$")
    if np.isfinite(tail_last):
        txt.append(rf"$Q_{{last}}={tail_last:.3e}$")
    if np.isfinite(r_tail):
        txt.append(rf"$R_{{tail}}={r_tail:.3e}$")
    if np.isfinite(frac_nonincreasing):
        txt.append(rf"$f_{{noninc}}={frac_nonincreasing:.2f}$")

    if txt:
        ax.text(
            0.03, 0.97, "\n".join(txt),
            transform=ax.transAxes,
            va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
        )

    # ---- IMPORTANT FIX: linear x-axis and controlled ticks ----
    ax.set_xlim(box_kxd[0], box_kxd[1])
    ax.set_xticks([100, 120, 140, 160, 180])
    ax.set_xlabel(r"$k_x\delta$")
    ax.set_ylabel(rf"${field_label}$")
    ax.set_ylim(bottom=0.0)

    ax.set_title(
        rf"Boxed high-$k$ decay diagnostic"
        "\n"
        rf"$d^+\in[{box_dplus[0]:g},{box_dplus[1]:g}]$, "
        rf"$k_x\delta\in[{box_kxd[0]:g},{box_kxd[1]:g}]$"
    )

    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=260)
    plt.close(fig)

    with open(out_txt, "w") as f:
        f.write("=== BOXED HIGH-k DECAY DIAGNOSTIC ===\n")
        f.write(f"field_label: {field_label}\n")
        f.write(f"box_kxd: {box_kxd}\n")
        f.write(f"box_dplus: {box_dplus}\n")
        f.write(f"tail_kmin: {tail_kmin}\n")
        f.write(f"zero_threshold: {zero_threshold}\n")
        f.write(f"n_k_points_in_box: {np.sum(mk)}\n")
        f.write(f"n_dplus_bins_in_box: {np.sum(md)}\n")
        f.write(f"dplus_bins_used: {D_box.tolist()}\n\n")

        f.write(f"peak_mean_curve: {peak_mean_curve}\n")
        f.write(f"tail_mean: {tail_mean}\n")
        f.write(f"tail_last: {tail_last}\n")
        f.write(f"fraction_nonincreasing_in_tail: {frac_nonincreasing}\n")
        f.write(f"R_tail = <Q>_(kx*delta in [{box_kxd[0]},{box_kxd[1]}]) / max_k <Q> = {r_tail}\n\n")

        if np.isfinite(tail_last):
            if tail_last <= zero_threshold:
                f.write("INTERPRETATION: largest resolved high-k value is at a near-zero level.\n")
            else:
                f.write("INTERPRETATION: largest resolved high-k value remains above the chosen near-zero threshold.\n")

        if np.isfinite(r_tail):
            if r_tail < 0.05:
                f.write("INTERPRETATION: tail energy is very small relative to peak energy.\n")
            elif r_tail < 0.15:
                f.write("INTERPRETATION: tail energy is reduced but still visible.\n")
            else:
                f.write("INTERPRETATION: tail energy remains noticeably elevated relative to the peak.\n")

        f.write("NOTE: monotonicity is only a trend indicator; finite-resolution spectra need not be strictly monotone pointwise.\n")
        f.write("\nK_box:\n")
        f.write(" ".join(f"{v:.10e}" for v in K_box) + "\n")
        f.write("\nmean_curve:\n")
        f.write(" ".join(f"{v:.10e}" for v in mean_curve) + "\n")
        f.write("\nmedian_curve:\n")
        f.write(" ".join(f"{v:.10e}" for v in median_curve) + "\n")

    print("Saved:", out_png)
    print("Saved:", out_txt)

    return tail_mean, tail_last, frac_nonincreasing, r_tail


# ================= MAIN =================
def main():
    if u_tau <= 0.0 or nu <= 0.0:
        raise RuntimeError("u_tau and nu must be positive.")

    moser_style()

    tdirs = timeDirs
    if tdirs is None:
        cand = sorted([d for d in glob.glob(os.path.join(caseDir, time_glob)) if os.path.isdir(d)])
        cand = [os.path.basename(d) for d in cand]

        def is_float(s):
            try:
                float(s)
                return True
            except Exception:
                return False

        cand = [c for c in cand if is_float(c)]
        tdirs = cand[:max_times] if max_times is not None else cand
        if not tdirs:
            raise RuntimeError("No time directories found.")

    print("Times:", tdirs)

    Cpath = find_field_path(caseDir, tdirs[0], Cname)
    if Cpath is None:
        raise RuntimeError("Could not locate C field.")
    print("Reading C:", Cpath)
    C = read_foam_vector_field(Cpath)
    Ncells = C.shape[0]
    print("Ncells:", Ncells)

    V = None
    if useV:
        Vpath = find_field_path(caseDir, tdirs[0], Vname)
        if Vpath is None:
            print("WARNING: V not found, using uniform weights.")
        else:
            print("Reading V:", Vpath)
            V = read_foam_scalar_field(Vpath)

    Vw = np.ones(Ncells, dtype=np.float64) if V is None else V.astype(np.float64)

    x = C[:, 0]
    y = C[:, 1]
    z = C[:, 2]

    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    zmin, zmax = float(z.min()), float(z.max())

    Lx = float(Lx_user) if Lx_user is not None else (xmax - xmin)
    Ly = float(Ly_user) if Ly_user is not None else (ymax - ymin)
    Lz = float(Lz_user) if Lz_user is not None else (zmax - zmin)
    dlt = (Ly / 2.0) if (delta is None) else float(delta)

    print(f"Lx,Ly,Lz = {Lx:.6e} {Ly:.6e} {Lz:.6e} ; delta={dlt:.6e}")

    Nx, x_levels, x_idx, groups, meta = build_x_planes_and_lines(
        C, Vw, Nx_expected, tol_y=tol_y, tol_z=tol_z
    )

    print(f"Detected Nx planes: {Nx}")
    print(f"Quantization tolerances: tol_x={meta['tol_x']:.3e}, tol_y={meta['tol_y']:.3e}, tol_z={meta['tol_z']:.3e}")
    print(f"Number of (y,z)-lines: {groups['uniq_keys'].size}")

    dy_wall = np.minimum(y - ymin, ymax - y)
    dz_wall = np.minimum(z - zmin, zmax - z)
    dy_wall = np.maximum(dy_wall, 0.0)
    dz_wall = np.maximum(dz_wall, 0.0)
    d_wall = np.minimum(dy_wall, dz_wall)

    dplus_cell = d_wall * (u_tau / nu)

    order = groups["order"]
    starts = groups["starts"]
    counts = groups["counts"]

    nlines = starts.size
    line_dplus = np.zeros(nlines, dtype=np.float64)
    line_w = np.zeros(nlines, dtype=np.float64)

    for i in range(nlines):
        sl = order[starts[i]:starts[i] + counts[i]]
        w = Vw[sl]
        ws = np.sum(w)
        if ws <= 0:
            w = np.ones(sl.size)
            ws = float(sl.size)

        line_dplus[i] = np.sum(w * dplus_cell[sl]) / ws
        line_w[i] = ws

    d_edges = np.logspace(np.log10(dplus_min), np.log10(dplus_max), Nd_bins + 1)
    d_cent = np.sqrt(d_edges[:-1] * d_edges[1:])

    Nk = Nx // 2 + 1
    m = np.arange(Nk, dtype=np.float64)
    kx = (2.0 * np.pi / Lx) * m
    dkx = 2.0 * np.pi / Lx
    kxd = kx * dlt
    fac = one_sided_factor(Nx)

    Phi_uu_acc = np.zeros((Nd_bins, Nk), dtype=np.float64)
    Phi_vv_acc = np.zeros((Nd_bins, Nk), dtype=np.float64)
    Phi_ww_acc = np.zeros((Nd_bins, Nk), dtype=np.float64)

    uu_phys_acc = np.zeros(Nd_bins, dtype=np.float64)
    vv_phys_acc = np.zeros(Nd_bins, dtype=np.float64)
    ww_phys_acc = np.zeros(Nd_bins, dtype=np.float64)

    uu_spec_acc = np.zeros(Nd_bins, dtype=np.float64)
    vv_spec_acc = np.zeros(Nd_bins, dtype=np.float64)
    ww_spec_acc = np.zeros(Nd_bins, dtype=np.float64)

    bin_nt_acc = np.zeros(Nd_bins, dtype=np.int64)

    group_cellid = []
    group_xidx = []
    for i in range(nlines):
        sl = order[starts[i]:starts[i] + counts[i]]
        group_cellid.append(sl)
        group_xidx.append(x_idx[sl])

    nt = 0
    nlines_used = 0
    nlines_bad = 0

    for tdir in tdirs:
        Upath = os.path.join(caseDir, tdir, uPrimeName)
        if not os.path.isfile(Upath):
            print("Missing:", Upath, "-> skipping")
            continue

        print("\nTime:", tdir, "reading:", Upath)
        U = read_foam_vector_field(Upath).astype(np.float64)

        up = U[:, 0]
        vp = U[:, 1]
        wp = U[:, 2]

        Phi_uu_t = np.zeros((Nd_bins, Nk), dtype=np.float64)
        Phi_vv_t = np.zeros((Nd_bins, Nk), dtype=np.float64)
        Phi_ww_t = np.zeros((Nd_bins, Nk), dtype=np.float64)

        uu_phys_t = np.zeros(Nd_bins, dtype=np.float64)
        vv_phys_t = np.zeros(Nd_bins, dtype=np.float64)
        ww_phys_t = np.zeros(Nd_bins, dtype=np.float64)

        uu_spec_t = np.zeros(Nd_bins, dtype=np.float64)
        vv_spec_t = np.zeros(Nd_bins, dtype=np.float64)
        ww_spec_t = np.zeros(Nd_bins, dtype=np.float64)

        wbin_sum = np.zeros(Nd_bins, dtype=np.float64)

        used_this = 0
        bad_this = 0

        for i in range(nlines):
            dpl = line_dplus[i]
            b = np.searchsorted(d_edges, dpl) - 1
            if b < 0 or b >= Nd_bins:
                continue

            sl = group_cellid[i]
            xi = group_xidx[i]
            w_line = line_w[i]
            if w_line <= 0:
                continue

            uline = np.zeros(Nx, dtype=np.float64)
            vline = np.zeros(Nx, dtype=np.float64)
            wline = np.zeros(Nx, dtype=np.float64)
            wpx = np.zeros(Nx, dtype=np.float64)

            Vi = Vw[sl]
            for j, cid in enumerate(sl):
                k = int(xi[j])
                wk = float(Vi[j])
                uline[k] += wk * up[cid]
                vline[k] += wk * vp[cid]
                wline[k] += wk * wp[cid]
                wpx[k] += wk

            if not np.all(wpx > 0):
                bad_this += 1
                continue

            uline /= wpx
            vline /= wpx
            wline /= wpx

            uline -= np.mean(uline)
            vline -= np.mean(vline)
            wline -= np.mean(wline)

            Uhat = np.fft.rfft(uline) / float(Nx)
            Vhat = np.fft.rfft(vline) / float(Nx)
            What = np.fft.rfft(wline) / float(Nx)

            Phi_uu_line = (fac * (np.abs(Uhat) ** 2)) / dkx
            Phi_vv_line = (fac * (np.abs(Vhat) ** 2)) / dkx
            Phi_ww_line = (fac * (np.abs(What) ** 2)) / dkx

            uu_phys_line = np.mean(uline ** 2)
            vv_phys_line = np.mean(vline ** 2)
            ww_phys_line = np.mean(wline ** 2)

            uu_spec_line = np.sum(Phi_uu_line * dkx)
            vv_spec_line = np.sum(Phi_vv_line * dkx)
            ww_spec_line = np.sum(Phi_ww_line * dkx)

            Phi_uu_t[b, :] += w_line * Phi_uu_line
            Phi_vv_t[b, :] += w_line * Phi_vv_line
            Phi_ww_t[b, :] += w_line * Phi_ww_line

            uu_phys_t[b] += w_line * uu_phys_line
            vv_phys_t[b] += w_line * vv_phys_line
            ww_phys_t[b] += w_line * ww_phys_line

            uu_spec_t[b] += w_line * uu_spec_line
            vv_spec_t[b] += w_line * vv_spec_line
            ww_spec_t[b] += w_line * ww_spec_line

            wbin_sum[b] += w_line
            used_this += 1

        for b in range(Nd_bins):
            if wbin_sum[b] > 0:
                Phi_uu_t[b, :] /= wbin_sum[b]
                Phi_vv_t[b, :] /= wbin_sum[b]
                Phi_ww_t[b, :] /= wbin_sum[b]

                uu_phys_t[b] /= wbin_sum[b]
                vv_phys_t[b] /= wbin_sum[b]
                ww_phys_t[b] /= wbin_sum[b]

                uu_spec_t[b] /= wbin_sum[b]
                vv_spec_t[b] /= wbin_sum[b]
                ww_spec_t[b] /= wbin_sum[b]

        bin_present = (wbin_sum > 0)

        Phi_uu_acc[bin_present, :] += Phi_uu_t[bin_present, :]
        Phi_vv_acc[bin_present, :] += Phi_vv_t[bin_present, :]
        Phi_ww_acc[bin_present, :] += Phi_ww_t[bin_present, :]

        uu_phys_acc[bin_present] += uu_phys_t[bin_present]
        vv_phys_acc[bin_present] += vv_phys_t[bin_present]
        ww_phys_acc[bin_present] += ww_phys_t[bin_present]

        uu_spec_acc[bin_present] += uu_spec_t[bin_present]
        vv_spec_acc[bin_present] += vv_spec_t[bin_present]
        ww_spec_acc[bin_present] += ww_spec_t[bin_present]

        bin_nt_acc[bin_present] += 1

        nt += 1
        nlines_used += used_this
        nlines_bad += bad_this
        print(f"Lines used this snapshot: {used_this}, bad/incomplete: {bad_this}")

    if nt == 0:
        raise RuntimeError("No valid snapshots processed.")

    denom2 = np.maximum(bin_nt_acc, 1)[:, None].astype(np.float64)
    denom1 = np.maximum(bin_nt_acc, 1).astype(np.float64)

    Phi_uu = Phi_uu_acc / denom2
    Phi_vv = Phi_vv_acc / denom2
    Phi_ww = Phi_ww_acc / denom2

    uu_phys = uu_phys_acc / denom1
    vv_phys = vv_phys_acc / denom1
    ww_phys = ww_phys_acc / denom1

    uu_spec = uu_spec_acc / denom1
    vv_spec = vv_spec_acc / denom1
    ww_spec = ww_spec_acc / denom1

    never = (bin_nt_acc == 0)
    Phi_uu[never, :] = np.nan
    Phi_vv[never, :] = np.nan
    Phi_ww[never, :] = np.nan
    uu_phys[never] = np.nan
    vv_phys[never] = np.nan
    ww_phys[never] = np.nan
    uu_spec[never] = np.nan
    vv_spec[never] = np.nan
    ww_spec[never] = np.nan

    Quu = (kx[None, :] * Phi_uu) / (u_tau ** 2)
    Qvv = (kx[None, :] * Phi_vv) / (u_tau ** 2)
    Qww = (kx[None, :] * Phi_ww) / (u_tau ** 2)

    Phi_tot = Phi_uu + Phi_vv + Phi_ww
    Qtot = (kx[None, :] * Phi_tot) / (u_tau ** 2)

    if smooth_k_plot and smooth_k_plot > 1:
        for b in range(Nd_bins):
            for Q in (Quu, Qvv, Qww, Qtot):
                mask = np.isfinite(Q[b, :])
                if not np.any(mask):
                    continue
                tmp = np.zeros_like(Q[b, :])
                tmp[mask] = Q[b, mask]
                sm = moving_average_1d(tmp, smooth_k_plot)
                out = np.full_like(Q[b, :], np.nan)
                out[mask] = sm[mask]
                Q[b, :] = out

    pref = os.path.join(outDir, f"BOXTAIL_ONLY_Nx{Nx}_nt{nt}")

    box_tail_mean = np.nan
    box_tail_last = np.nan
    box_frac_noninc = np.nan
    box_r_tail = np.nan

    if ENABLE_BOX_DIAGNOSTIC:
        if BOX_FIELD == "uu":
            Qbox = Quu
            field_label = r"k_x\Phi_{uu}/u_\tau^2"
            tag = "uu"
        elif BOX_FIELD == "vv":
            Qbox = Qvv
            field_label = r"k_x\Phi_{vv}/u_\tau^2"
            tag = "vv"
        elif BOX_FIELD == "ww":
            Qbox = Qww
            field_label = r"k_x\Phi_{ww}/u_\tau^2"
            tag = "ww"
        elif BOX_FIELD == "tot":
            Qbox = Qtot
            field_label = r"k_x(\Phi_{uu}+\Phi_{vv}+\Phi_{ww})/u_\tau^2"
            tag = "tot"
        else:
            raise RuntimeError(f"Unknown BOX_FIELD = {BOX_FIELD}")

        box_tail_mean, box_tail_last, box_frac_noninc, box_r_tail = boxed_tail_diagnostic(
            kxd=kxd,
            d_cent=d_cent,
            Q=Qbox,
            bin_nt=bin_nt_acc,
            out_png=pref + f"_BOXCHECK_{tag}_tail_decay.png",
            out_txt=pref + f"_BOXCHECK_{tag}_tail_decay.txt",
            field_label=field_label,
            box_kxd=BOX_KXD,
            box_dplus=BOX_DPLUS,
            tail_kmin=TAIL_CHECK_KXD_MIN,
            zero_threshold=TAIL_ZERO_THRESHOLD,
        )

    rel_uu = np.abs(uu_spec - uu_phys) / np.maximum(np.abs(uu_phys), 1e-300)
    rel_vv = np.abs(vv_spec - vv_phys) / np.maximum(np.abs(vv_phys), 1e-300)
    rel_ww = np.abs(ww_spec - ww_phys) / np.maximum(np.abs(ww_phys), 1e-300)

    rep = pref + "_VALIDATION.txt"
    with open(rep, "w") as f:
        f.write("=== BOXED HIGH-k TAIL DIAGNOSTIC VALIDATION ===\n")
        f.write(f"caseDir: {caseDir}\n")
        f.write(f"times requested: {tdirs}\n")
        f.write(f"snapshots processed: {nt}\n")
        f.write(f"Nx detected: {Nx}\n")
        f.write(f"Lx,Ly,Lz: {Lx:.15e} {Ly:.15e} {Lz:.15e}\n")
        f.write(f"u_tau: {u_tau:.15e}\n")
        f.write(f"nu: {nu:.15e}\n")
        f.write(f"delta: {dlt:.15e}\n")
        f.write(f"tol_x,tol_y,tol_z: {meta['tol_x']:.3e} {meta['tol_y']:.3e} {meta['tol_z']:.3e}\n")
        f.write(f"Lines used total: {nlines_used}\n")
        f.write(f"Bad/incomplete total: {nlines_bad}\n")
        f.write("\n")

        f.write("--- Parseval check per nearest-wall d+ bin ---\n")
        f.write("dplus    bin_nt   uu_phys   uu_spec   rel_uu    vv_phys   vv_spec   rel_vv    ww_phys   ww_spec   rel_ww\n")
        for b in range(Nd_bins):
            f.write(
                f"{d_cent[b]:10.4e}  {bin_nt_acc[b]:6d}  "
                f"{uu_phys[b]:10.4e} {uu_spec[b]:10.4e} {rel_uu[b]:9.2e}  "
                f"{vv_phys[b]:10.4e} {vv_spec[b]:10.4e} {rel_vv[b]:9.2e}  "
                f"{ww_phys[b]:10.4e} {ww_spec[b]:10.4e} {rel_ww[b]:9.2e}\n"
            )

        f.write("\n--- Box settings ---\n")
        f.write(f"BOX_FIELD: {BOX_FIELD}\n")
        f.write(f"BOX_KXD: {BOX_KXD}\n")
        f.write(f"BOX_DPLUS: {BOX_DPLUS}\n")
        f.write(f"TAIL_CHECK_KXD_MIN: {TAIL_CHECK_KXD_MIN}\n")
        f.write(f"TAIL_ZERO_THRESHOLD: {TAIL_ZERO_THRESHOLD}\n")

        f.write("\n--- Box outputs ---\n")
        f.write(f"box_tail_mean: {box_tail_mean}\n")
        f.write(f"box_tail_last: {box_tail_last}\n")
        f.write(f"box_frac_noninc: {box_frac_noninc}\n")
        f.write(f"R_tail: {box_r_tail}\n")

    print("Saved:", rep)

    if save_npz:
        np.savez(
            pref + "_spectra.npz",
            kx=kx,
            kxd=kxd,
            dplus=d_cent,
            Phi_uu=Phi_uu,
            Phi_vv=Phi_vv,
            Phi_ww=Phi_ww,
            Quu=Quu,
            Qvv=Qvv,
            Qww=Qww,
            Phi_tot=Phi_tot,
            Qtot=Qtot,
            bin_nt_acc=bin_nt_acc,
            uu_phys=uu_phys,
            uu_spec=uu_spec,
            rel_uu=rel_uu,
            vv_phys=vv_phys,
            vv_spec=vv_spec,
            rel_vv=rel_vv,
            ww_phys=ww_phys,
            ww_spec=ww_spec,
            rel_ww=rel_ww,
            Lx=Lx,
            Ly=Ly,
            Lz=Lz,
            u_tau=u_tau,
            nu=nu,
            delta=dlt,
            nt=nt,
            BOX_FIELD=np.array([BOX_FIELD], dtype=object),
            BOX_KXD=np.array(BOX_KXD, dtype=np.float64),
            BOX_DPLUS=np.array(BOX_DPLUS, dtype=np.float64),
            TAIL_CHECK_KXD_MIN=TAIL_CHECK_KXD_MIN,
            TAIL_ZERO_THRESHOLD=TAIL_ZERO_THRESHOLD,
            box_tail_mean=box_tail_mean,
            box_tail_last=box_tail_last,
            box_frac_noninc=box_frac_noninc,
            box_r_tail=box_r_tail,
        )
        print("Saved:", pref + "_spectra.npz")


if __name__ == "__main__":
    main()