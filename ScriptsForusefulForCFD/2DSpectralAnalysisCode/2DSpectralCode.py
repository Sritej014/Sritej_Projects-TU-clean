#!/usr/bin/env python3
# ============================================================
# x-only (streamwise) 1D spectra on UNSTRUCTURED / multiblock mesh:
#   - x periodic, sampled on regular planes  -> FFT in x
#   - y,z wall-bounded                       -> no FFT in y,z
#   - anisotropy preserved                   -> compute uu, vv, ww separately
#
# Key idea for unstructured mesh:
#   Group cells into "x-lines" by quantizing (y,z) -> each group contains Nx samples in x.
#   Then FFT along x for each (y,z)-line, and average spectra over cross-section with V weights.
#
# Outputs:
#  (1) Contour maps (d+): log10(kx*Phi_uu/u_tau^2), log10(kx*Phi_vv/u_tau^2), log10(kx*Phi_ww/u_tau^2)
#  (2) Curves at multiple d+: kx*Phi_ii/u_tau^2 vs kx*delta
#  (3) Split-family anisotropy maps:
#        - y-wall family: log10(Phi_vv/Phi_uu) vs d_y+
#        - z-wall family: log10(Phi_ww/Phi_uu) vs d_z+
#  (4) Full-domain energy distribution maps (NEW):
#        - log10(kx*(Phi_uu+Phi_vv+Phi_ww)/u_tau^2)
#        - component fractions: Phi_uu/Phi_tot, Phi_vv/Phi_tot, Phi_ww/Phi_tot
#  (5) Inertial diagnostics per timestamp + overlay + time-mean
#  (6) Parseval validation per d+ bin
#
# IMPORTANT FIXES:
#   - bin-wise snapshot averaging (no zero-bin bias)
#   - split wall-family anisotropy to avoid nearest-wall mixing artifacts
#   - low-energy masking in anisotropy ratio maps
#   - occupancy diagnostics for d+ bins
#
# COMMON COLORBAR SUPPORT (NEW):
#   - Set fixed log10 color limits per quantity (Quu/Qvv/Qww/Qtot)
#   - Ensures apples-to-apples comparison across CFL / nCFL / cyclic
# ============================================================

import os, re, glob, mmap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- USER SETTINGS ----------------
caseDir   = "/WORK/sritej/TASKFILES/Task1/FOR_EXPERIMENT/planar"
timeDirs = [
    "0.81295453", "0.81669453", "0.82043453", "0.82417453", "0.82791453"
]
time_glob = "[0-9]*"
max_times = 50

# Expected Nx along x
Nx_expected = 372

# OpenFOAM fields
uPrimeName = "subtract(U,UMean)"   # vector field of fluctuations (u',v',w')
Cname      = "C"                   # cell centers
Vname      = "V"                   # cell volumes
useV       = True

# Domain lengths (if known; else inferred from C)
Lx_user = 1.860000000000e-01
Ly_user = 6.200000000000e-02
Lz_user = 6.200000000000e-02

# Wall scaling
u_tau = 0.855
nu    = 18.4e-06

# delta for kx*delta scaling (default Ly/2)
delta = None

# Grouping tolerances in y and z (auto if None)
tol_y = None
tol_z = None

# d+ binning
dplus_min = 0.5
dplus_max = 3000.0
Nd_bins   = 35  # reduced from 60 to reduce striping from sparse bins

# Curves at selected d+ (nearest bin)
dplus_curves = [3, 5, 8, 12, 20, 40, 80, 160, 320, 640]

# Slope diagnostics
slope_band_dplus = (90.0, 170.0)
inertial_fit_kxd = (8.0, 25.0)

# Optional smoothing for PLOTS ONLY
smooth_k_plot = 5  # 0 disables; odd recommended

# Ratio-map masking (for anisotropy noise suppression)
ratio_energy_floor_rel = 1e-6
ratio_min_abs = 1e-20

# Outputs
outDir   = os.getcwd()
save_npz = True

# ---------------- COMMON COLORBAR SETTINGS (NEW) ----------------
# IMPORTANT:
# - These limits are in LOG10 space AFTER premultiplication, i.e. for log10(kx*Phi/u_tau^2)
# - To compare CFL/nCFL/cyclic fairly, use the SAME limits for all compared runs.
# - If a value is None, that plot falls back to automatic percentile scaling.
# - Suggested workflow:
#     1) Decide these limits once (e.g., from cyclic reference or union of NPZs)
#     2) Reuse same values across all runs
USE_COMMON_COLORBARS = True
NLEVELS_LOG = 18

COMMON_LOG_LIMITS = {
    # Examples (edit as needed). None = auto percentile scaling
    "Quu":  None,   # e.g. (-2.8, 0.6)
    "Qvv":  None,   # e.g. (-3.2, 0.2)
    "Qww":  None,   # e.g. (-3.2, 0.2)
    "Qtot": None,   # e.g. (-2.8, 0.8)
}

# Fraction maps are already common by construction [0,1]
NLEVELS_FRACTION = 17

# Optional fixed symmetric range for anisotropy ratio maps (log10 ratios).
# Set to None for auto percentile; set e.g. 1.0 for common ±1 range.
COMMON_RATIO_MX = {
    "YWALL": None,
    "ZWALL": None,
}


# ---------------- OpenFOAM ASCII readers ----------------
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
        raise RuntimeError("Could not parse N and list '(' after nonuniform header")
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


# ---------------- plotting helpers ----------------
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
    Nk = Nx//2 + 1
    fac = np.ones(Nk, dtype=np.float64)
    if Nk > 2:
        fac[1:-1] = 2.0
    return fac


# ---------------- core unstructured grouping ----------------
def infer_tol_from_levels(coord):
    s = np.sort(coord.astype(np.float64))
    ds = np.diff(s)
    ds = ds[ds > 0]
    if ds.size == 0:
        return 1e-9
    dmin = np.percentile(ds, 1.0)
    return 0.25 * dmin

def build_x_planes_and_lines(C, Vw, Nx_expected, tol_y=None, tol_z=None):
    x = C[:,0].astype(np.float64)
    y = C[:,1].astype(np.float64)
    z = C[:,2].astype(np.float64)

    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    zmin, zmax = float(z.min()), float(z.max())

    # detect x planes
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
            f"If this is intentional, set Nx_expected=None and re-run."
        )

    # quantize y,z for x-lines
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

    # representative yz and weights
    y_rep = np.zeros(uniq_keys.size, dtype=np.float64)
    z_rep = np.zeros(uniq_keys.size, dtype=np.float64)
    w_rep = np.zeros(uniq_keys.size, dtype=np.float64)

    for i in range(uniq_keys.size):
        sl = order[starts[i]:starts[i]+counts[i]]
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
        "tol_x": tol_x, "tol_y": tol_y, "tol_z": tol_z
    }
    groups = {
        "order": order,
        "uniq_keys": uniq_keys,
        "starts": starts,
        "counts": counts,
        "y_rep": y_rep,
        "z_rep": z_rep,
        "w_rep": w_rep
    }
    return Nx, x_levels, x_idx, groups, meta


# ---------------- MAIN ----------------
def main():
    if u_tau <= 0 or nu <= 0:
        raise RuntimeError("Set positive u_tau and nu.")

    moser_style()

    # resolve times
    tdirs = timeDirs
    if tdirs is None:
        cand = sorted([d for d in glob.glob(os.path.join(caseDir, time_glob)) if os.path.isdir(d)])
        cand = [os.path.basename(d) for d in cand]
        def is_float(s):
            try:
                float(s)
                return True
            except:
                return False
        cand = [c for c in cand if is_float(c)]
        tdirs = cand[:max_times] if max_times is not None else cand
        if not tdirs:
            raise RuntimeError("No time directories found.")
    print("Times:", tdirs)

    # read C
    Cpath = find_field_path(caseDir, tdirs[0], Cname)
    if Cpath is None:
        raise RuntimeError("Could not locate C field.")
    print("Reading C:", Cpath)
    C = read_foam_vector_field(Cpath)
    Ncells = C.shape[0]
    print("Ncells:", Ncells)

    # read V
    V = None
    if useV:
        Vpath = find_field_path(caseDir, tdirs[0], Vname)
        if Vpath is None:
            print("WARNING: V not found -> using uniform weights.")
        else:
            print("Reading V:", Vpath)
            V = read_foam_scalar_field(Vpath)

    Vw = np.ones(Ncells, dtype=np.float64) if V is None else V.astype(np.float64)

    # infer domain lengths
    x = C[:,0]
    y = C[:,1]
    z = C[:,2]
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    zmin, zmax = float(z.min()), float(z.max())

    Lx = float(Lx_user) if Lx_user is not None else (xmax - xmin)
    Ly = float(Ly_user) if Ly_user is not None else (ymax - ymin)
    Lz = float(Lz_user) if Lz_user is not None else (zmax - zmin)
    dlt = (Ly / 2.0) if (delta is None) else float(delta)

    print(f"Lx,Ly,Lz = {Lx:.6e} {Ly:.6e} {Lz:.6e} ; delta={dlt:.6e}")

    # build x-planes and x-lines
    Nx, x_levels, x_idx, groups, meta = build_x_planes_and_lines(
        C, Vw, Nx_expected, tol_y=tol_y, tol_z=tol_z
    )
    print(f"Detected Nx planes: {Nx}")
    print(f"Quantization tolerances: tol_x={meta['tol_x']:.3e}, tol_y={meta['tol_y']:.3e}, tol_z={meta['tol_z']:.3e}")
    print(f"Number of (y,z)-lines: {groups['uniq_keys'].size}")

    pref_base = os.path.join(outDir, f"UNSTRUCT_XSPECTRA_yzWALLS_Nx{Nx}")

    # --- Wall distances (split family + nearest-wall legacy) ---
    dy_wall = np.minimum(y - ymin, ymax - y)
    dz_wall = np.minimum(z - zmin, zmax - z)
    dy_wall = np.maximum(dy_wall, 0.0)
    dz_wall = np.maximum(dz_wall, 0.0)

    d_wall = np.minimum(dy_wall, dz_wall)  # legacy nearest wall for generic maps

    dplus_cell   = d_wall * (u_tau / nu)
    dplus_y_cell = dy_wall * (u_tau / nu)
    dplus_z_cell = dz_wall * (u_tau / nu)

    order = groups["order"]
    starts = groups["starts"]
    counts = groups["counts"]

    # per-line representative d+ and line weight
    nlines = starts.size
    line_dplus   = np.zeros(nlines, dtype=np.float64)  # nearest wall
    line_dplus_y = np.zeros(nlines, dtype=np.float64)  # y-wall family
    line_dplus_z = np.zeros(nlines, dtype=np.float64)  # z-wall family
    line_w       = np.zeros(nlines, dtype=np.float64)

    for i in range(nlines):
        sl = order[starts[i]:starts[i] + counts[i]]
        w = Vw[sl]
        ws = np.sum(w)
        if ws <= 0:
            w = np.ones(sl.size)
            ws = float(sl.size)

        line_dplus[i]   = np.sum(w * dplus_cell[sl]) / ws
        line_dplus_y[i] = np.sum(w * dplus_y_cell[sl]) / ws
        line_dplus_z[i] = np.sum(w * dplus_z_cell[sl]) / ws
        line_w[i] = ws

    # d+ bins
    d_edges = np.logspace(np.log10(dplus_min), np.log10(dplus_max), Nd_bins + 1)
    d_cent  = np.sqrt(d_edges[:-1] * d_edges[1:])

    # k grid
    Nk = Nx // 2 + 1
    m = np.arange(Nk, dtype=np.float64)
    kx = (2.0 * np.pi / Lx) * m
    dkx = 2.0 * np.pi / Lx
    kxd = kx * dlt
    fac = one_sided_factor(Nx)

    # --- Accumulators (nearest-wall generic maps / curves / validation) ---
    Phi_uu_acc = np.zeros((Nd_bins, Nk), dtype=np.float64)
    Phi_vv_acc = np.zeros((Nd_bins, Nk), dtype=np.float64)
    Phi_ww_acc = np.zeros((Nd_bins, Nk), dtype=np.float64)

    uu_phys_acc = np.zeros(Nd_bins, dtype=np.float64)
    vv_phys_acc = np.zeros(Nd_bins, dtype=np.float64)
    ww_phys_acc = np.zeros(Nd_bins, dtype=np.float64)

    uu_spec_acc = np.zeros(Nd_bins, dtype=np.float64)
    vv_spec_acc = np.zeros(Nd_bins, dtype=np.float64)
    ww_spec_acc = np.zeros(Nd_bins, dtype=np.float64)

    bin_nt_acc  = np.zeros(Nd_bins, dtype=np.int64)     # snapshots contributing per nearest-wall bin
    bin_w_acc   = np.zeros(Nd_bins, dtype=np.float64)   # time-summed line weights (diagnostic)

    # --- Split-family anisotropy accumulators ---
    # y-wall family (vv/uu) vs d_y+
    Phi_uu_y_acc = np.zeros((Nd_bins, Nk), dtype=np.float64)
    Phi_vv_y_acc = np.zeros((Nd_bins, Nk), dtype=np.float64)
    bin_nt_y_acc = np.zeros(Nd_bins, dtype=np.int64)
    bin_w_y_acc  = np.zeros(Nd_bins, dtype=np.float64)

    # z-wall family (ww/uu) vs d_z+
    Phi_uu_z_acc = np.zeros((Nd_bins, Nk), dtype=np.float64)
    Phi_ww_z_acc = np.zeros((Nd_bins, Nk), dtype=np.float64)
    bin_nt_z_acc = np.zeros(Nd_bins, dtype=np.int64)
    bin_w_z_acc  = np.zeros(Nd_bins, dtype=np.float64)

    nt = 0
    nlines_used = 0
    nlines_bad  = 0

    # precompute per-line cell ids and x-plane ids
    group_cellid = []
    group_xidx   = []
    for i in range(nlines):
        sl = order[starts[i]:starts[i] + counts[i]]
        group_cellid.append(sl)
        group_xidx.append(x_idx[sl])

    # per-timestamp inertial diagnostics storage
    band_times = []
    band_Phiuu = []
    band_slope = []
    band_deltaslope = []

    for tdir in tdirs:
        Upath = os.path.join(caseDir, tdir, uPrimeName)
        if not os.path.isfile(Upath):
            print("Missing:", Upath, " -> skipping")
            continue

        print("\nTime:", tdir, "reading:", Upath)
        U = read_foam_vector_field(Upath).astype(np.float64)
        up = U[:, 0]
        vp = U[:, 1]
        wp = U[:, 2]

        # nearest-wall per-snapshot accumulators
        Phi_uu_t = np.zeros((Nd_bins, Nk), dtype=np.float64)
        Phi_vv_t = np.zeros((Nd_bins, Nk), dtype=np.float64)
        Phi_ww_t = np.zeros((Nd_bins, Nk), dtype=np.float64)

        uu_phys_t = np.zeros(Nd_bins, dtype=np.float64)
        vv_phys_t = np.zeros(Nd_bins, dtype=np.float64)
        ww_phys_t = np.zeros(Nd_bins, dtype=np.float64)

        uu_spec_t = np.zeros(Nd_bins, dtype=np.float64)
        vv_spec_t = np.zeros(Nd_bins, dtype=np.float64)
        ww_spec_t = np.zeros(Nd_bins, dtype=np.float64)

        wbin_sum  = np.zeros(Nd_bins, dtype=np.float64)

        # split-family per-snapshot accumulators
        Phi_uu_y_t = np.zeros((Nd_bins, Nk), dtype=np.float64)
        Phi_vv_y_t = np.zeros((Nd_bins, Nk), dtype=np.float64)
        wbin_y_sum = np.zeros(Nd_bins, dtype=np.float64)
        nbin_y_lines_t = np.zeros(Nd_bins, dtype=np.int64)

        Phi_uu_z_t = np.zeros((Nd_bins, Nk), dtype=np.float64)
        Phi_ww_z_t = np.zeros((Nd_bins, Nk), dtype=np.float64)
        wbin_z_sum = np.zeros(Nd_bins, dtype=np.float64)
        nbin_z_lines_t = np.zeros(Nd_bins, dtype=np.int64)

        used_this = 0
        bad_this  = 0

        for i in range(nlines):
            # nearest-wall bin for generic spectra maps / curves / validation
            dpl = line_dplus[i]
            b = np.searchsorted(d_edges, dpl) - 1
            if b < 0 or b >= Nd_bins:
                continue

            sl = group_cellid[i]
            xi = group_xidx[i]
            w_line = line_w[i]
            if w_line <= 0:
                continue

            # Build x-line arrays by plane index with volume averaging
            uline = np.zeros(Nx, dtype=np.float64)
            vline = np.zeros(Nx, dtype=np.float64)
            wline = np.zeros(Nx, dtype=np.float64)
            wpx   = np.zeros(Nx, dtype=np.float64)

            Vi = Vw[sl]
            for j, cid in enumerate(sl):
                k = int(xi[j])
                wk = float(Vi[j])
                uline[k] += wk * up[cid]
                vline[k] += wk * vp[cid]
                wline[k] += wk * wp[cid]
                wpx[k]   += wk

            if not np.all(wpx > 0):
                bad_this += 1
                continue

            uline /= wpx
            vline /= wpx
            wline /= wpx

            # remove x-mean (kills k=0)
            uline -= np.mean(uline)
            vline -= np.mean(vline)
            wline -= np.mean(wline)

            # FFT in x with û=(1/Nx)FFT
            Uhat = np.fft.rfft(uline) / float(Nx)
            Vhat = np.fft.rfft(vline) / float(Nx)
            What = np.fft.rfft(wline) / float(Nx)

            # Phi = (one-sided |û|^2) / dk
            Phi_uu_line = (fac * (np.abs(Uhat) ** 2)) / dkx
            Phi_vv_line = (fac * (np.abs(Vhat) ** 2)) / dkx
            Phi_ww_line = (fac * (np.abs(What) ** 2)) / dkx

            # physical variances
            uu_phys_line = np.mean(uline ** 2)
            vv_phys_line = np.mean(vline ** 2)
            ww_phys_line = np.mean(wline ** 2)

            # spectral variances
            uu_spec_line = np.sum(Phi_uu_line * dkx)
            vv_spec_line = np.sum(Phi_vv_line * dkx)
            ww_spec_line = np.sum(Phi_ww_line * dkx)

            # nearest-wall accumulation (generic maps)
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

            # split-family anisotropy accumulation
            # y-wall family -> vv/uu vs d_y+
            by = np.searchsorted(d_edges, line_dplus_y[i]) - 1
            if 0 <= by < Nd_bins:
                Phi_uu_y_t[by, :] += w_line * Phi_uu_line
                Phi_vv_y_t[by, :] += w_line * Phi_vv_line
                wbin_y_sum[by]    += w_line
                nbin_y_lines_t[by] += 1

            # z-wall family -> ww/uu vs d_z+
            bz = np.searchsorted(d_edges, line_dplus_z[i]) - 1
            if 0 <= bz < Nd_bins:
                Phi_uu_z_t[bz, :] += w_line * Phi_uu_line
                Phi_ww_z_t[bz, :] += w_line * Phi_ww_line
                wbin_z_sum[bz]    += w_line
                nbin_z_lines_t[bz] += 1

            used_this += 1

        # normalize nearest-wall snapshot bin averages
        for b in range(Nd_bins):
            if wbin_sum[b] > 0:
                Phi_uu_t[b, :] /= wbin_sum[b]
                Phi_vv_t[b, :] /= wbin_sum[b]
                Phi_ww_t[b, :] /= wbin_sum[b]
                uu_phys_t[b]   /= wbin_sum[b]
                vv_phys_t[b]   /= wbin_sum[b]
                ww_phys_t[b]   /= wbin_sum[b]
                uu_spec_t[b]   /= wbin_sum[b]
                vv_spec_t[b]   /= wbin_sum[b]
                ww_spec_t[b]   /= wbin_sum[b]

        # normalize split-family snapshot bin averages
        for b in range(Nd_bins):
            if wbin_y_sum[b] > 0:
                Phi_uu_y_t[b, :] /= wbin_y_sum[b]
                Phi_vv_y_t[b, :] /= wbin_y_sum[b]
            if wbin_z_sum[b] > 0:
                Phi_uu_z_t[b, :] /= wbin_z_sum[b]
                Phi_ww_z_t[b, :] /= wbin_z_sum[b]

        # per-timestamp inertial diagnostics (band-averaged Phi_uu over nearest-wall bins)
        dlo, dhi = slope_band_dplus
        band_mask = (d_cent >= dlo) & (d_cent <= dhi)
        band_present = band_mask & (wbin_sum > 0)

        slope_fit_t = np.nan
        delta_slope_t = np.nan

        if np.any(band_present):
            wband = wbin_sum[band_present]
            Phi_band_t = np.sum(Phi_uu_t[band_present, :] * wband[:, None], axis=0) / np.sum(wband)

            kmin, kmax = inertial_fit_kxd
            mfit = (kxd >= kmin) & (kxd <= kmax) & (m > 0) & (Phi_band_t > 0)

            if np.sum(mfit) >= 6:
                p = np.polyfit(np.log10(kx[mfit]), np.log10(Phi_band_t[mfit]), 1)
                slope_fit_t = float(p[0])
                delta_slope_t = slope_fit_t - (-5.0 / 3.0)

            pref_t = pref_base + f"_t{tdir}"

            # (A) compensated plot
            fig = plt.figure(figsize=(6.2, 5.0))
            comp_t = (kx[1:] ** (5.0 / 3.0)) * Phi_band_t[1:]
            plt.loglog(kxd[1:], comp_t, lw=1.6)
            plt.xlabel(r"$k_x \delta$")
            plt.ylabel(r"$k_x^{5/3}\Phi_{uu}(k_x)$")
            plt.title(rf"Compensated: $k_x^{{5/3}}\Phi_{{uu}}$  (t={tdir})")
            if np.isfinite(slope_fit_t):
                plt.text(0.05, 0.06,
                         f"slope fit = {slope_fit_t:.3f}\ndelta vs -5/3 = {delta_slope_t:+.3f}\nfit: [{kmin:g},{kmax:g}]",
                         transform=plt.gca().transAxes)
            plt.tight_layout()
            plt.savefig(pref_t + "_compensated_k53_Phiuu.png", dpi=260)
            plt.close(fig)

            # (B) Phi + -5/3 ref
            fig = plt.figure(figsize=(6.2, 5.0))
            plt.loglog(kxd[1:], Phi_band_t[1:], lw=1.6, label=r"$\Phi_{uu}$ (band avg)")
            if np.isfinite(slope_fit_t):
                kref = np.sqrt(kmin * kmax)
                iref = int(np.argmin(np.abs(kxd - kref)))
                yref = Phi_band_t[iref]
                kline = np.array([kmin, kmax], dtype=np.float64)
                yline = yref * ((kline / kref) ** (-5.0 / 3.0))
                plt.loglog(kline, yline, "--", lw=1.4, label=r"$k^{-5/3}$ ref")
                plt.text(0.05, 0.06,
                         f"fit slope = {slope_fit_t:.3f}\ndelta vs -5/3 = {delta_slope_t:+.3f}\nfit: [{kmin:g},{kmax:g}]",
                         transform=plt.gca().transAxes)
            plt.xlabel(r"$k_x \delta$")
            plt.ylabel(r"$\Phi_{uu}(k_x)$")
            plt.title(rf"Slope fit on $\Phi_{{uu}}$ (t={tdir})")
            plt.legend(frameon=False)
            plt.tight_layout()
            plt.savefig(pref_t + "_inertial_slope_Phiuu.png", dpi=260)
            plt.close(fig)

            band_times.append(tdir)
            band_Phiuu.append(Phi_band_t.copy())
            band_slope.append(slope_fit_t)
            band_deltaslope.append(delta_slope_t)

        # accumulate nearest-wall snapshot means (only bins present)
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
        bin_w_acc += wbin_sum  # diagnostic

        # accumulate split-family snapshot means
        present_y = (wbin_y_sum > 0)
        present_z = (wbin_z_sum > 0)

        Phi_uu_y_acc[present_y, :] += Phi_uu_y_t[present_y, :]
        Phi_vv_y_acc[present_y, :] += Phi_vv_y_t[present_y, :]
        bin_nt_y_acc[present_y]    += 1
        bin_w_y_acc                += wbin_y_sum

        Phi_uu_z_acc[present_z, :] += Phi_uu_z_t[present_z, :]
        Phi_ww_z_acc[present_z, :] += Phi_ww_z_t[present_z, :]
        bin_nt_z_acc[present_z]    += 1
        bin_w_z_acc                += wbin_z_sum

        nt += 1
        nlines_used += used_this
        nlines_bad  += bad_this
        print(f"Lines used this snapshot: {used_this}, bad/incomplete: {bad_this}")

    if nt == 0:
        raise RuntimeError("No valid time dirs processed.")

    print(f"\nProcessed snapshots: {nt}")
    print(f"Total lines used: {nlines_used} ; total bad/incomplete lines: {nlines_bad}")

    # --- Finalize nearest-wall time means (bin-wise) ---
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

    # --- Finalize split-family time means ---
    den_y = np.maximum(bin_nt_y_acc, 1)[:, None].astype(np.float64)
    den_z = np.maximum(bin_nt_z_acc, 1)[:, None].astype(np.float64)

    Phi_uu_y = Phi_uu_y_acc / den_y
    Phi_vv_y = Phi_vv_y_acc / den_y
    Phi_uu_z = Phi_uu_z_acc / den_z
    Phi_ww_z = Phi_ww_z_acc / den_z

    never_y = (bin_nt_y_acc == 0)
    never_z = (bin_nt_z_acc == 0)

    Phi_uu_y[never_y, :] = np.nan
    Phi_vv_y[never_y, :] = np.nan
    Phi_uu_z[never_z, :] = np.nan
    Phi_ww_z[never_z, :] = np.nan

    # premultiplied spectra (nearest-wall maps)
    Quu = (kx[None, :] * Phi_uu) / (u_tau ** 2)
    Qvv = (kx[None, :] * Phi_vv) / (u_tau ** 2)
    Qww = (kx[None, :] * Phi_ww) / (u_tau ** 2)

    # NEW: full-domain total energy distribution map
    Phi_tot = Phi_uu + Phi_vv + Phi_ww
    Qtot = (kx[None, :] * Phi_tot) / (u_tau ** 2)

    # NEW: component energy fractions (raw Phi fractions)
    eps_frac = 1e-300
    Fuu = Phi_uu / np.maximum(Phi_tot, eps_frac)
    Fvv = Phi_vv / np.maximum(Phi_tot, eps_frac)
    Fww = Phi_ww / np.maximum(Phi_tot, eps_frac)

    # smoothing for display only (nearest-wall premultiplied maps)
    if smooth_k_plot and smooth_k_plot > 1:
        for b in range(Nd_bins):
            if np.all(~np.isfinite(Quu[b, :])):
                continue
            for Q in (Quu, Qvv, Qww, Qtot):
                mask = np.isfinite(Q[b, :])
                tmp = np.zeros_like(Q[b, :])
                tmp[mask] = Q[b, mask]
                sm = moving_average_1d(tmp, smooth_k_plot)
                out = np.full_like(Q[b, :], np.nan)
                out[mask] = sm[mask]
                Q[b, :] = out

    pref = os.path.join(outDir, f"UNSTRUCT_XSPECTRA_yzWALLS_Nx{Nx}_nt{nt}")

    # ---------------- overlay plots across timestamps ----------------
    if len(band_Phiuu) > 0:
        kmin, kmax = inertial_fit_kxd

        fig = plt.figure(figsize=(6.2, 5.0))
        for tdir, P, s in zip(band_times, band_Phiuu, band_slope):
            comp = (kx[1:] ** (5.0 / 3.0)) * P[1:]
            lab = f"t={tdir}"
            if np.isfinite(s):
                lab += f" (s={s:.2f})"
            plt.loglog(kxd[1:], comp, lw=1.1, label=lab)
        plt.xlabel(r"$k_x \delta$")
        plt.ylabel(r"$k_x^{5/3}\Phi_{uu}(k_x)$")
        plt.title(r"Compensated $k_x^{5/3}\Phi_{uu}$ (each timestamp)")
        plt.legend(frameon=False, fontsize=8)
        plt.tight_layout()
        plt.savefig(pref + "_ALLTIMES_compensated_k53_Phiuu.png", dpi=260)
        plt.close(fig)

        fig = plt.figure(figsize=(6.2, 5.0))
        for tdir, P, s in zip(band_times, band_Phiuu, band_slope):
            lab = f"t={tdir}"
            if np.isfinite(s):
                lab += f" (s={s:.2f})"
            plt.loglog(kxd[1:], P[1:], lw=1.1, label=lab)

        kref = np.sqrt(kmin * kmax)
        iref = int(np.argmin(np.abs(kxd - kref)))
        Pref = band_Phiuu[0]
        yref = Pref[iref]
        kline = np.array([kmin, kmax], dtype=np.float64)
        yline = yref * ((kline / kref) ** (-5.0 / 3.0))
        plt.loglog(kline, yline, "--", lw=1.6, label=r"$k^{-5/3}$ ref")

        plt.xlabel(r"$k_x \delta$")
        plt.ylabel(r"$\Phi_{uu}(k_x)$")
        plt.title(r"$\Phi_{uu}(k_x)$ (each timestamp) + $k^{-5/3}$ ref")
        plt.legend(frameon=False, fontsize=8)
        plt.tight_layout()
        plt.savefig(pref + "_ALLTIMES_inertial_slope_Phiuu.png", dpi=260)
        plt.close(fig)

    # ---------------- plotting helpers (use local K,D) ----------------
    K = kxd[1:]  # skip k=0
    D = d_cent
    eps = 1e-12

    # Track actual auto limits used (useful for reporting/debug)
    auto_log_ranges_used = {}

    def contour_log10(Q, outname, title, cbar_label=None, clim=None, nlevels=18, cmap="Reds", keyname=None):
        """
        Plot log10 contour of premultiplied spectrum Q with optional common color limits.

        Parameters
        ----------
        Q : 2D array [Nd_bins, Nk]
        clim : tuple(lo, hi) in LOG10 space, or None for auto percentile scaling
        keyname : optional label for storing/printing the used range
        """
        A = Q[:, 1:]  # skip k=0
        Z = np.log10(np.maximum(A, eps))
        finite = np.isfinite(Z)
        if not np.any(finite):
            print("WARNING: contour_log10 has no finite data for", outname)
            return

        if clim is None:
            lo = float(np.percentile(Z[finite], 5.0))
            hi = float(np.percentile(Z[finite], 99.0))
            used_mode = "AUTO"
        else:
            lo, hi = float(clim[0]), float(clim[1])
            used_mode = "COMMON"

        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            raise RuntimeError(f"Invalid contour limits for {outname}: lo={lo}, hi={hi}")

        levels = np.linspace(lo, hi, int(nlevels))

        if keyname is not None:
            auto_log_ranges_used[keyname] = (lo, hi, used_mode)

        fig = plt.figure(figsize=(6.4, 4.8))
        KK, DD = np.meshgrid(K, D)
        cf = plt.contourf(KK, DD, Z, levels=levels, cmap=cmap, extend="both")
        plt.contour(KK, DD, Z, levels=levels[::2], colors="k", linewidths=0.7)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlim(max(K.min(), 1e-1), min(K.max(), 1e3))
        plt.ylim(max(D.min(), dplus_min), min(D.max(), dplus_max))
        plt.xlabel(r"$k_x \delta$")
        plt.ylabel(r"$d^+$ (nearest wall)")
        plt.title(title)
        cbar = plt.colorbar(cf)
        cbar.set_label(cbar_label if cbar_label else r"$\log_{10}(k_x\Phi/u_\tau^2)$")
        plt.tight_layout()
        plt.savefig(outname, dpi=260)
        plt.close(fig)

        print(f"[{used_mode}] {keyname if keyname else outname}: log10 range = ({lo:.3f}, {hi:.3f})")

    def contour_ratio(Pnum, Pden, outname, title, ylab=r"$d^+$", bin_counts=None, bin_weights=None, mx_fixed=None):
        A_num = Pnum[:, 1:].copy()
        A_den = Pden[:, 1:].copy()

        finite_num = np.isfinite(A_num)
        finite_den = np.isfinite(A_den)
        if not np.any(finite_num) or not np.any(finite_den):
            print("WARNING: contour_ratio has no finite data for", outname)
            return

        num_max = np.nanmax(A_num[finite_num]) if np.any(finite_num) else 0.0
        den_max = np.nanmax(A_den[finite_den]) if np.any(finite_den) else 0.0

        num_thr = max(ratio_min_abs, ratio_energy_floor_rel * max(num_max, ratio_min_abs))
        den_thr = max(ratio_min_abs, ratio_energy_floor_rel * max(den_max, ratio_min_abs))

        valid = np.isfinite(A_num) & np.isfinite(A_den) & (A_num > num_thr) & (A_den > den_thr)

        Z = np.full_like(A_num, np.nan, dtype=np.float64)
        Z[valid] = np.log10(A_num[valid] / A_den[valid])

        finite = np.isfinite(Z)
        if not np.any(finite):
            print("WARNING: contour_ratio all masked for", outname)
            return

        if mx_fixed is None:
            lo = np.percentile(Z[finite], 2.0)
            hi = np.percentile(Z[finite], 98.0)
            mx = max(abs(lo), abs(hi))
            used_mode = "AUTO"
        else:
            mx = float(mx_fixed)
            used_mode = "COMMON"

        levels = np.linspace(-mx, mx, 19)

        fig = plt.figure(figsize=(6.8, 5.0))
        KK, DD = np.meshgrid(K, D)
        cf = plt.contourf(KK, DD, Z, levels=levels, cmap="RdBu_r", extend="both")
        plt.contour(KK, DD, Z, levels=levels[::2], colors="k", linewidths=0.6)

        plt.xscale("log")
        plt.yscale("log")
        plt.xlim(max(K.min(), 1e-1), min(K.max(), 1e3))
        plt.ylim(max(D.min(), dplus_min), min(D.max(), dplus_max))
        plt.xlabel(r"$k_x \delta$")
        plt.ylabel(ylab)
        plt.title(title)

        cbar = plt.colorbar(cf)
        cbar.set_label(r"$\log_{10}(\Phi_{ii}/\Phi_{uu})$")

        # Occupancy summary text
        txt = [f"ratio range mode: {used_mode}"]
        if bin_counts is not None:
            bc = np.asarray(bin_counts)
            txt.append(f"nonempty bins: {np.sum(bc>0)}/{len(bc)}")
            txt.append(f"bin_nt min/max: {int(np.min(bc))}/{int(np.max(bc))}")
        if bin_weights is not None:
            bw = np.asarray(bin_weights)
            pos = bw[bw > 0]
            if pos.size > 0:
                txt.append(f"binW min/max: {np.min(pos):.2e}/{np.max(pos):.2e}")
        if txt:
            plt.text(1.02, 0.02, "\n".join(txt), transform=plt.gca().transAxes,
                     va="bottom", ha="left", fontsize=9)

        plt.tight_layout()
        plt.savefig(outname, dpi=260)
        plt.close(fig)

    def contour_fraction(F, outname, title, ylab=r"$d^+$ (nearest wall)"):
        A = F[:, 1:]  # skip k=0
        Z = np.clip(A, 0.0, 1.0)
        finite = np.isfinite(Z)
        if not np.any(finite):
            print("WARNING: contour_fraction has no finite data for", outname)
            return

        levels = np.linspace(0.0, 1.0, NLEVELS_FRACTION)
        fig = plt.figure(figsize=(6.4, 4.8))
        KK, DD = np.meshgrid(K, D)
        cf = plt.contourf(KK, DD, Z, levels=levels, cmap="viridis", extend="neither")
        plt.contour(KK, DD, Z, levels=levels[::2], colors="k", linewidths=0.5)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlim(max(K.min(), 1e-1), min(K.max(), 1e3))
        plt.ylim(max(D.min(), dplus_min), min(D.max(), dplus_max))
        plt.xlabel(r"$k_x \delta$")
        plt.ylabel(ylab)
        plt.title(title)
        cbar = plt.colorbar(cf)
        cbar.set_label("Energy fraction of total")
        plt.tight_layout()
        plt.savefig(outname, dpi=260)
        plt.close(fig)

    def plot_bin_occupancy(bin_counts, bin_weights, outname, title):
        fig = plt.figure(figsize=(6.2, 4.8))
        ax1 = plt.gca()
        ax1.semilogy(d_cent, np.maximum(bin_counts, 1), "o-", lw=1.4, ms=4)
        ax1.set_xlabel(r"$d^+$ bin center")
        ax1.set_ylabel("snapshot count per bin (bin_nt)")
        ax1.set_title(title)
        ax1.grid(True, which="both", alpha=0.25)

        if bin_weights is not None:
            ax2 = ax1.twinx()
            bw = np.array(bin_weights, dtype=np.float64)
            bw_plot = np.where(bw > 0, bw, np.nan)
            ax2.semilogy(d_cent, bw_plot, "s--", lw=1.1, ms=3)
            ax2.set_ylabel("time-summed line weight per bin")

        plt.tight_layout()
        plt.savefig(outname, dpi=260)
        plt.close(fig)

    # ---------------- plots ----------------

    # Helper to resolve common-vs-auto limits
    def _get_common_clim(key):
        if not USE_COMMON_COLORBARS:
            return None
        val = COMMON_LOG_LIMITS.get(key, None)
        return val if val is not None else None

    # (1) Component premultiplied energy maps (nearest-wall coordinate)
    contour_log10(
        Quu,
        pref + "_contour_log_kxPhiuu_dplus.png",
        r"$\log_{10}(k_x\Phi_{uu}/u_\tau^2)$  (x-only, yz wall-bounded)",
        clim=_get_common_clim("Quu"),
        nlevels=NLEVELS_LOG,
        keyname="Quu"
    )
    contour_log10(
        Qvv,
        pref + "_contour_log_kxPhivv_dplus.png",
        r"$\log_{10}(k_x\Phi_{vv}/u_\tau^2)$  (x-only, yz wall-bounded)",
        clim=_get_common_clim("Qvv"),
        nlevels=NLEVELS_LOG,
        keyname="Qvv"
    )
    contour_log10(
        Qww,
        pref + "_contour_log_kxPhiww_dplus.png",
        r"$\log_{10}(k_x\Phi_{ww}/u_\tau^2)$  (x-only, yz wall-bounded)",
        clim=_get_common_clim("Qww"),
        nlevels=NLEVELS_LOG,
        keyname="Qww"
    )

    # (1c) NEW: full-domain total energy distribution map (covers whole d+ and kxd region)
    contour_log10(
        Qtot,
        pref + "_FULLDOMAIN_contour_log_kxPhiTOTAL_dplus.png",
        r"$\log_{10}(k_x(\Phi_{uu}+\Phi_{vv}+\Phi_{ww})/u_\tau^2)$  (full-domain energy distribution)",
        clim=_get_common_clim("Qtot"),
        nlevels=NLEVELS_LOG,
        keyname="Qtot"
    )

    # (1d) NEW: component fractions of total spectral energy (already common [0,1])
    contour_fraction(
        Fuu,
        pref + "_FULLDOMAIN_fraction_Phiuu_over_Phitot.png",
        r"$\Phi_{uu}/(\Phi_{uu}+\Phi_{vv}+\Phi_{ww})$"
    )
    contour_fraction(
        Fvv,
        pref + "_FULLDOMAIN_fraction_Phivv_over_Phitot.png",
        r"$\Phi_{vv}/(\Phi_{uu}+\Phi_{vv}+\Phi_{ww})$"
    )
    contour_fraction(
        Fww,
        pref + "_FULLDOMAIN_fraction_Phiww_over_Phitot.png",
        r"$\Phi_{ww}/(\Phi_{uu}+\Phi_{vv}+\Phi_{ww})$"
    )

    # (2) Split-family anisotropy maps (recommended for duct-like cross section)
    contour_ratio(
        Phi_vv_y, Phi_uu_y,
        pref + "_anisotropy_YWALL_log10_PhiVV_over_PhiUU.png",
        r"$\log_{10}(\Phi_{vv}/\Phi_{uu})$ vs $d_y^+$  (y-wall family only)",
        ylab=r"$d_y^+$",
        bin_counts=bin_nt_y_acc,
        bin_weights=bin_w_y_acc,
        mx_fixed=COMMON_RATIO_MX["YWALL"]
    )

    contour_ratio(
        Phi_ww_z, Phi_uu_z,
        pref + "_anisotropy_ZWALL_log10_PhiWW_over_PhiUU.png",
        r"$\log_{10}(\Phi_{ww}/\Phi_{uu})$ vs $d_z^+$  (z-wall family only)",
        ylab=r"$d_z^+$",
        bin_counts=bin_nt_z_acc,
        bin_weights=bin_w_z_acc,
        mx_fixed=COMMON_RATIO_MX["ZWALL"]
    )

    # Optional legacy nearest-wall anisotropy maps for debugging (commented)
    # contour_ratio(Phi_vv, Phi_uu,
    #     pref + "_anisotropy_legacy_nearestwall_vv_over_uu.png",
    #     r"$\log_{10}(\Phi_{vv}/\Phi_{uu})$ vs nearest-wall $d^+$ (legacy)",
    #     ylab=r"$d^+$", bin_counts=bin_nt_acc, bin_weights=bin_w_acc)
    # contour_ratio(Phi_ww, Phi_uu,
    #     pref + "_anisotropy_legacy_nearestwall_ww_over_uu.png",
    #     r"$\log_{10}(\Phi_{ww}/\Phi_{uu})$ vs nearest-wall $d^+$ (legacy)",
    #     ylab=r"$d^+$", bin_counts=bin_nt_acc, bin_weights=bin_w_acc)

    # (3) Curves at multiple d+ for uu,vv,ww (nearest-wall bins)
    def plot_curves(Q, tag):
        fig = plt.figure(figsize=(6.2, 5.0))
        for dreq in dplus_curves:
            b = int(np.argmin(np.abs(d_cent - dreq)))
            if np.all(~np.isfinite(Q[b, 1:])):
                continue
            plt.semilogx(kxd[1:], Q[b, 1:], lw=1.4, label=rf"$d^+\approx{d_cent[b]:.0f}$")
        plt.xlim(max(kxd[1], 1e-1), min(kxd[-1], 1e3))
        plt.ylim(bottom=0.0)
        plt.xlabel(r"$k_x \delta$")
        plt.ylabel(rf"$k_x \Phi_{{{tag}{tag}}}/u_\tau^2$")
        plt.title(rf"$k_x \Phi_{{{tag}{tag}}}/u_\tau^2$ at multiple $d^+$ (x-only)")
        plt.legend(frameon=False, fontsize=9)
        plt.tight_layout()
        plt.savefig(pref + f"_curves_kxPhi{tag}{tag}_many_dplus.png", dpi=260)
        plt.close(fig)

    plot_curves(Quu, "u")
    plot_curves(Qvv, "v")
    plot_curves(Qww, "w")

    # (4) Inertial diagnostics on TIME-MEAN Phi_uu averaged over d+ band (nearest-wall bins)
    dlo, dhi = slope_band_dplus
    band = (d_cent >= dlo) & (d_cent <= dhi) & (bin_nt_acc > 0)
    slope_fit = np.nan
    delta_slope = np.nan

    if np.any(band):
        Phi_band = np.nanmean(Phi_uu[band, :], axis=0)

        fig = plt.figure(figsize=(6.2, 5.0))
        comp = (kx[1:] ** (5.0 / 3.0)) * Phi_band[1:]
        plt.loglog(kxd[1:], comp, lw=1.6)
        plt.xlabel(r"$k_x \delta$")
        plt.ylabel(r"$k_x^{5/3}\Phi_{uu}(k_x)$")
        plt.title(r"Compensated inertial check (time-mean): $k_x^{5/3}\Phi_{uu}$")
        plt.tight_layout()
        plt.savefig(pref + "_compensated_k53_Phiuu.png", dpi=260)
        plt.close(fig)

        kmin, kmax = inertial_fit_kxd
        mfit = (kxd >= kmin) & (kxd <= kmax) & (m > 0) & (Phi_band > 0)
        if np.sum(mfit) >= 6:
            p = np.polyfit(np.log10(kx[mfit]), np.log10(Phi_band[mfit]), 1)
            slope_fit = float(p[0])
            delta_slope = slope_fit - (-5.0 / 3.0)

        fig = plt.figure(figsize=(6.2, 5.0))
        plt.loglog(kxd[1:], Phi_band[1:], lw=1.6, label=r"$\Phi_{uu}$ (band avg)")
        if np.isfinite(slope_fit):
            kref = np.sqrt(kmin * kmax)
            iref = int(np.argmin(np.abs(kxd - kref)))
            yref = Phi_band[iref]
            kline = np.array([kmin, kmax], dtype=np.float64)
            yline = yref * ((kline / kref) ** (-5.0 / 3.0))
            plt.loglog(kline, yline, "--", lw=1.4, label=r"$k^{-5/3}$ ref")
            plt.text(0.05, 0.06,
                     f"fit slope = {slope_fit:.3f}\ndelta vs -5/3 = {delta_slope:+.3f}\nfit band: [{kmin:g},{kmax:g}]",
                     transform=plt.gca().transAxes)
        plt.xlabel(r"$k_x \delta$")
        plt.ylabel(r"$\Phi_{uu}(k_x)$")
        plt.title(r"Slope fit on $\Phi_{uu}$ (time-mean)")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(pref + "_inertial_slope_Phiuu.png", dpi=260)
        plt.close(fig)

    # (5) Occupancy diagnostic plots (VERY useful to explain striping / blank bands)
    plot_bin_occupancy(
        bin_nt_acc, bin_w_acc,
        pref + "_occupancy_nearestwall_bins.png",
        "Nearest-wall d+ bin occupancy (generic maps)"
    )
    plot_bin_occupancy(
        bin_nt_y_acc, bin_w_y_acc,
        pref + "_occupancy_YWALL_anisotropy_bins.png",
        "y-wall family anisotropy bin occupancy"
    )
    plot_bin_occupancy(
        bin_nt_z_acc, bin_w_z_acc,
        pref + "_occupancy_ZWALL_anisotropy_bins.png",
        "z-wall family anisotropy bin occupancy"
    )

    # (6) Parseval validation per d+ bin (time-mean nearest-wall)
    rel_uu = np.abs(uu_spec - uu_phys) / np.maximum(np.abs(uu_phys), 1e-300)
    rel_vv = np.abs(vv_spec - vv_phys) / np.maximum(np.abs(vv_phys), 1e-300)
    rel_ww = np.abs(ww_spec - ww_phys) / np.maximum(np.abs(ww_phys), 1e-300)

    rep = pref + "_VALIDATION.txt"
    with open(rep, "w") as f:
        f.write("=== x-only spectra validation (x periodic, y/z wall-bounded, unstructured/multiblock mesh) ===\n")
        f.write(f"caseDir: {caseDir}\n")
        f.write(f"times requested: {tdirs}\n")
        f.write(f"snapshots processed (nt): {nt}\n")
        f.write(f"Nx planes detected: {Nx}\n")
        f.write(f"Lx,Ly,Lz: {Lx:.15e} {Ly:.15e} {Lz:.15e}\n")
        f.write(f"u_tau, nu: {u_tau:.15e} {nu:.15e}\n")
        f.write(f"delta: {dlt:.15e}\n")
        f.write(f"tol_x,tol_y,tol_z: {meta['tol_x']:.3e} {meta['tol_y']:.3e} {meta['tol_z']:.3e}\n")
        f.write(f"d+ bins: [{dplus_min},{dplus_max}] Nd={Nd_bins}\n")
        f.write(f"smooth_k_plot: {smooth_k_plot}\n")
        f.write(f"ratio_energy_floor_rel: {ratio_energy_floor_rel}\n")
        f.write(f"ratio_min_abs: {ratio_min_abs}\n")
        f.write(f"Lines used total: {nlines_used} ; bad/incomplete: {nlines_bad}\n\n")

        f.write("--- Common colorbar settings ---\n")
        f.write(f"USE_COMMON_COLORBARS: {USE_COMMON_COLORBARS}\n")
        f.write(f"NLEVELS_LOG: {NLEVELS_LOG}\n")
        f.write(f"COMMON_LOG_LIMITS: {COMMON_LOG_LIMITS}\n")
        f.write(f"COMMON_RATIO_MX: {COMMON_RATIO_MX}\n")
        f.write("Actual ranges used (log10):\n")
        for kname, (lo, hi, mode) in auto_log_ranges_used.items():
            f.write(f"  {kname}: ({lo:.6f}, {hi:.6f}) [{mode}]\n")
        f.write("\n")

        f.write("--- Parseval check per nearest-wall d+ bin (time-mean) ---\n")
        f.write("dplus    bin_nt   uu_phys   uu_spec   rel_uu    vv_phys   vv_spec   rel_vv    ww_phys   ww_spec   rel_ww\n")
        for b in range(Nd_bins):
            f.write(f"{d_cent[b]:10.4e}  {bin_nt_acc[b]:6d}  "
                    f"{uu_phys[b]:10.4e} {uu_spec[b]:10.4e} {rel_uu[b]:9.2e}  "
                    f"{vv_phys[b]:10.4e} {vv_spec[b]:10.4e} {rel_vv[b]:9.2e}  "
                    f"{ww_phys[b]:10.4e} {ww_spec[b]:10.4e} {rel_ww[b]:9.2e}\n")

        f.write("\n--- Inertial diagnostics (uu, time-mean, nearest-wall bins) ---\n")
        f.write(f"d+ band: {slope_band_dplus}\n")
        f.write(f"kx*delta fit band: {inertial_fit_kxd}\n")
        f.write(f"fit slope on Phi_uu: {slope_fit}\n")
        f.write(f"delta slope vs -5/3: {delta_slope}\n")

        f.write("\n--- Per-timestamp inertial slopes (band avg) ---\n")
        f.write("time        slope_fit     delta_vs_-5/3\n")
        for tdir, s, ds in zip(band_times, band_slope, band_deltaslope):
            f.write(f"{tdir:>10s}  {s:12.6f}  {ds:14.6f}\n")

        f.write("\n--- Occupancy summaries (split-family anisotropy) ---\n")
        f.write("Y-wall family: d_y+ bins used / total = "
                f"{np.sum(bin_nt_y_acc>0)}/{Nd_bins}, min/max bin_nt = {np.min(bin_nt_y_acc)}/{np.max(bin_nt_y_acc)}\n")
        f.write("Z-wall family: d_z+ bins used / total = "
                f"{np.sum(bin_nt_z_acc>0)}/{Nd_bins}, min/max bin_nt = {np.min(bin_nt_z_acc)}/{np.max(bin_nt_z_acc)}\n")

    # Print saved files summary
    print("Saved:", pref + "_contour_log_kxPhiuu_dplus.png")
    print("Saved:", pref + "_contour_log_kxPhivv_dplus.png")
    print("Saved:", pref + "_contour_log_kxPhiww_dplus.png")
    print("Saved:", pref + "_FULLDOMAIN_contour_log_kxPhiTOTAL_dplus.png")
    print("Saved:", pref + "_FULLDOMAIN_fraction_Phiuu_over_Phitot.png")
    print("Saved:", pref + "_FULLDOMAIN_fraction_Phivv_over_Phitot.png")
    print("Saved:", pref + "_FULLDOMAIN_fraction_Phiww_over_Phitot.png")
    print("Saved:", pref + "_anisotropy_YWALL_log10_PhiVV_over_PhiUU.png")
    print("Saved:", pref + "_anisotropy_ZWALL_log10_PhiWW_over_PhiUU.png")
    print("Saved:", pref + "_occupancy_nearestwall_bins.png")
    print("Saved:", pref + "_occupancy_YWALL_anisotropy_bins.png")
    print("Saved:", pref + "_occupancy_ZWALL_anisotropy_bins.png")
    print("Saved:", pref + "_curves_kxPhiuu_many_dplus.png")
    print("Saved:", pref + "_curves_kxPhivv_many_dplus.png")
    print("Saved:", pref + "_curves_kxPhiww_many_dplus.png")
    print("Saved:", pref + "_compensated_k53_Phiuu.png")
    print("Saved:", pref + "_inertial_slope_Phiuu.png")
    print("Saved:", rep)

    if save_npz:
        np.savez(
            pref + "_spectra.npz",
            kx=kx, kxd=kxd, dplus=d_cent,
            Phi_uu=Phi_uu, Phi_vv=Phi_vv, Phi_ww=Phi_ww,
            Quu=Quu, Qvv=Qvv, Qww=Qww,
            Phi_tot=Phi_tot, Qtot=Qtot,
            Fuu=Fuu, Fvv=Fvv, Fww=Fww,

            # split-family anisotropy arrays
            Phi_uu_y=Phi_uu_y, Phi_vv_y=Phi_vv_y,
            Phi_uu_z=Phi_uu_z, Phi_ww_z=Phi_ww_z,
            bin_nt_y_acc=bin_nt_y_acc, bin_nt_z_acc=bin_nt_z_acc,
            bin_w_y_acc=bin_w_y_acc, bin_w_z_acc=bin_w_z_acc,

            # nearest-wall occupancy
            bin_nt_acc=bin_nt_acc, bin_w_acc=bin_w_acc,

            # validation
            uu_phys=uu_phys, uu_spec=uu_spec, rel_uu=rel_uu,
            vv_phys=vv_phys, vv_spec=vv_spec, rel_vv=rel_vv,
            ww_phys=ww_phys, ww_spec=ww_spec, rel_ww=rel_ww,

            # metadata
            Lx=Lx, Ly=Ly, Lz=Lz, u_tau=u_tau, nu=nu, delta=dlt,
            dplus_min=dplus_min, dplus_max=dplus_max, Nd_bins=Nd_bins,
            slope_band_dplus=slope_band_dplus, inertial_fit_kxd=inertial_fit_kxd,
            slope_fit=slope_fit, delta_slope=delta_slope,
            ratio_energy_floor_rel=ratio_energy_floor_rel,
            ratio_min_abs=ratio_min_abs,
            nt=nt,
            band_times=np.array(band_times, dtype=object),
            band_slope=np.array(band_slope, dtype=np.float64),
            band_deltaslope=np.array(band_deltaslope, dtype=np.float64),

            # common colorbar metadata (for traceability)
            USE_COMMON_COLORBARS=USE_COMMON_COLORBARS,
            NLEVELS_LOG=NLEVELS_LOG,
            COMMON_LOG_LIMITS=np.array([str(COMMON_LOG_LIMITS)], dtype=object),
            COMMON_RATIO_MX=np.array([str(COMMON_RATIO_MX)], dtype=object),
        )
        print("Saved:", pref + "_spectra.npz")


if __name__ == "__main__":
    main()
