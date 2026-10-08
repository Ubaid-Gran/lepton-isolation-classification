"""W -> lnu analysis with ATLAS Open Data (13 TeV, 1lep skim).

Final version: channel selection, atomic verified downloads,
automatic cleanup of corrupted files, chunked reading, BDT.
"""
import os, glob, shutil, urllib.request
import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

os.makedirs("figures", exist_ok=True)

# ---------------- Configuration ----------------
LUMI = 10.0        # fb^-1
CHANNEL = "mu"     # "el"=W->enu, "mu"=W->munu, "both"=combined
QUICK_TEST = True  # True: minimal file set for fast first run

BASE   = ("https://opendata.cern.ch/eos/opendata/atlas/"
    "OutreachDatasets/2020-01-22/1lep/")
MIRROR = ("https://atlas-opendata.web.cern.ch/atlas-opendata/"
    "samples/2020/1lep/")

# ---------------- Samples (per channel) ----------------
W_FILES = {
    "el": ["MC/mc_361100.Wplusenu.1lep.root",
        "MC/mc_361103.Wminusenu.1lep.root"],
    "mu": ["MC/mc_361101.Wplusmunu.1lep.root",
        "MC/mc_361104.Wminusmunu.1lep.root"],
    }
W_FILES["both"] = W_FILES["el"] + W_FILES["mu"]

Z_FILES = {
    "el": ["MC/mc_361106.Zee.1lep.root"],
    "mu": ["MC/mc_361107.Zmumu.1lep.root"],
    }
Z_FILES["both"] = Z_FILES["el"] + Z_FILES["mu"]

SAMPLES = {
    "Data": {"files": [
            "Data/data_A.1lep.root", "Data/data_B.1lep.root",
            "Data/data_C.1lep.root", "Data/data_D.1lep.root"]},
    "W+jets":     {"label": 1, "files": W_FILES[CHANNEL]},
    "Z+jets":     {"label": 0, "files": Z_FILES[CHANNEL]},
    "ttbar":      {"label": 0,
        "files": ["MC/mc_410000.ttbar_lep.1lep.root"]},
    "Single top": {"label": 0, "files": [
            "MC/mc_410011.single_top_tchan.1lep.root",
            "MC/mc_410012.single_antitop_tchan.1lep.root"]},
    "Diboson":    {"label": 0, "files": [
            "MC/mc_363491.lllv.1lep.root",
            "MC/mc_363492.llvv.1lep.root"]},
    }

if QUICK_TEST:
    SAMPLES = {
        "Data":   {"files": ["Data/data_A.1lep.root"]},
        "W+jets": {"label": 1,
            "files": W_FILES[CHANNEL][:1]},
        "ttbar":  {"label": 0,
            "files": ["MC/mc_410000.ttbar_lep.1lep.root"]},
        }

BRANCHES = ["lep_n", "lep_pt", "lep_eta", "lep_phi",
    "lep_type", "lep_ptcone30", "lep_etcone20",
    "lep_isTightID", "met_et", "met_phi", "jet_n"]
MC_WEIGHTS = ["mcWeight", "scaleFactor_PILEUP",
    "scaleFactor_ELE", "scaleFactor_MUON",
    "scaleFactor_LepTRIGGER",
    "XSection", "SumWeights"]

# -------- Cleanup: remove corrupted leftovers --------
for f in glob.glob("*.part"):
    os.remove(f)
for f in glob.glob("*.root"):
    try:
        with uproot.open(f) as fh:
            fh["mini"].num_entries
    except Exception:
        print("Removing corrupted file:", f)
        os.remove(f)

# ---------------- Download helpers ----------------
def fetch(f):
    local = f.replace("/", "_")
    if os.path.exists(local):
        return local
    tmp = local + ".part"
    for base in (BASE, MIRROR):
        try:
            print("Downloading", base + f)
            urllib.request.urlretrieve(base + f, tmp)
            req = urllib.request.Request(
                base + f, method="HEAD")
            expected = int(urllib.request.urlopen(
                req).headers["Content-Length"])
            actual = os.path.getsize(tmp)
            if actual != expected:
                raise IOError(
                    f"incomplete: {actual}/{expected} bytes")
            os.rename(tmp, local)
            return local
        except Exception as e:
            print("  failed:", e)
            if os.path.exists(tmp):
                os.remove(tmp)
    raise RuntimeError(
        f"Could not download {f} from any server")

# ---------------- Event selection ----------------
def select(a, is_mc):
    a = a[a["lep_n"] == 1]
    if len(a) == 0:
        return None
    lep_pt_mev = ak.to_numpy(a["lep_pt"][:, 0])
    lep_pt   = lep_pt_mev / 1000.0
    lep_eta  = ak.to_numpy(a["lep_eta"][:, 0])
    lep_phi  = ak.to_numpy(a["lep_phi"][:, 0])
    lep_type = ak.to_numpy(a["lep_type"][:, 0])
    tight    = ak.to_numpy(
        a["lep_isTightID"][:, 0]).astype(bool)
    iso_trk  = ak.to_numpy(
        a["lep_ptcone30"][:, 0]) / lep_pt_mev
    iso_cal  = ak.to_numpy(
        a["lep_etcone20"][:, 0]) / lep_pt_mev
    met      = ak.to_numpy(a["met_et"]) / 1000.0
    met_phi  = ak.to_numpy(a["met_phi"])
    jet_n    = ak.to_numpy(a["jet_n"])

    dphi = np.abs(lep_phi - met_phi)
    dphi = np.where(dphi > np.pi,
        2.0 * np.pi - dphi, dphi)
    mt = np.sqrt(
        2.0 * lep_pt * met * (1.0 - np.cos(dphi)))

    if CHANNEL == "el":
        chan = (lep_type == 11)
    elif CHANNEL == "mu":
        chan = (lep_type == 13)
    else:
        chan = np.ones_like(lep_type, dtype=bool)

        cuts = (chan & (lep_pt > 35)
            & (np.abs(lep_eta) < 2.5) & tight
            & (iso_trk < 0.1) & (iso_cal < 0.1)
            & (met > 30) & (mt > 30))

        out = {"lep_pt":  lep_pt[cuts],
            "lep_eta": lep_eta[cuts],
            "iso_trk": iso_trk[cuts],
            "iso_cal": iso_cal[cuts],
            "met":     met[cuts],
            "mt":      mt[cuts],
            "jet_n":   jet_n[cuts]}

        if is_mc:
            w = (ak.to_numpy(a["mcWeight"])
                * ak.to_numpy(a["scaleFactor_PILEUP"])
                * ak.to_numpy(a["scaleFactor_ELE"])
                * ak.to_numpy(a["scaleFactor_MUON"])
                * ak.to_numpy(a["scaleFactor_LepTRIGGER"])
                * ak.to_numpy(a["XSection"])
                * 1000.0 * LUMI
                / ak.to_numpy(a["SumWeights"]))
            out["weight"] = w[cuts]
        else:
            out["weight"] = np.ones(int(cuts.sum()))
        return out

def load(name, info):
    is_mc    = name != "Data"
    branches = BRANCHES + (MC_WEIGHTS if is_mc else [])
    parts    = []
    for f in info["files"]:
        local = fetch(f)
        for chunk in uproot.iterate(
            local + ":mini", branches,
            step_size="200 MB", library="ak"):
            sel = select(chunk, is_mc)
            if sel is not None:
                parts.append(sel)
    return {k: np.concatenate([p[k] for p in parts])
        for k in parts[0]}

print(f"Loading samples (channel = {CHANNEL})...")
results = {name: load(name, info)
    for name, info in SAMPLES.items()}

# ----------- Data / MC stacked plots -----------
MC_ORDER = [s for s in
    ["Diboson", "Single top", "Z+jets",
        "ttbar", "W+jets"]
    if s in results]
COLORS = {"Diboson":    "purple",
    "Single top": "orange",
    "Z+jets":     "green",
    "ttbar":      "red",
    "W+jets":     "steelblue"}
CH_LABEL = {"el":   r"$W\to e\nu$",
    "mu":   r"$W\to\mu\nu$",
    "both": r"$W\to\ell\nu$"}[CHANNEL]
PLOTS = [
    ("lep_pt",  r"Lepton $p_T$ [GeV]",
        np.linspace(35, 200, 50),    "lep_pt"),
    ("met",     r"$E_T^{miss}$ [GeV]",
        np.linspace(30, 200, 50),    "met"),
    ("mt",      r"$m_T$ [GeV]",
        np.linspace(30, 180, 60),    "mt"),
    ("iso_trk", r"$p_T^{cone30}/p_T$",
        np.linspace(0, 0.1, 40),     "iso_track"),
    ("iso_cal", r"$E_T^{cone20}/p_T$",
        np.linspace(-0.05, 0.1, 40), "iso_calo"),
    ]

for var, xlabel, bins, fname in PLOTS:
    plt.figure(figsize=(7, 5))
    plt.hist(
        [results[s][var] for s in MC_ORDER],
        bins=bins, stacked=True,
        weights=[results[s]["weight"] for s in MC_ORDER],
        label=MC_ORDER,
        color=[COLORS[s] for s in MC_ORDER])
    counts, edges = np.histogram(
        results["Data"][var], bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    plt.errorbar(centers, counts, yerr=np.sqrt(counts),
        fmt="ko", markersize=3, label="Data")
    plt.xlabel(xlabel)
    plt.ylabel("Events")
    plt.yscale("log")
    plt.legend(fontsize=8)
    plt.title(CH_LABEL
        + r", ATLAS Open Data, $\sqrt{s}=13$ TeV,"
        r" 10 fb$^{-1}$")
    plt.tight_layout()
    plt.savefig(f"figures/{fname}_{CHANNEL}.png", dpi=200)
    plt.show(); plt.close()

# ---------------- Yields ----------------
print(f"\nEvent yields after selection "
    f"(channel = {CHANNEL}):")
for s in MC_ORDER + ["Data"]:
    print(f"  {s:12s} "
        f"{results[s]['weight'].sum():14.0f}")

# ---------------- BDT training ----------------
FEATURES = ["lep_pt", "lep_eta", "iso_trk", "iso_cal",
    "met", "mt", "jet_n"]
X = np.vstack(
    [np.column_stack([results[s][f] for f in FEATURES])
        for s in MC_ORDER])
y = np.concatenate(
    [np.full(len(results[s]["weight"]),
            SAMPLES[s]["label"])
        for s in MC_ORDER])
w = np.concatenate(
    [np.abs(results[s]["weight"]) for s in MC_ORDER])

X_tr, X_te, y_tr, y_te, w_tr, w_te = \
train_test_split(X, y, w, test_size=0.3,
    random_state=42, stratify=y)

bdt = XGBClassifier(n_estimators=400, max_depth=3,
    learning_rate=0.05, subsample=0.8,
    eval_metric="auc")
bdt.fit(X_tr, y_tr, sample_weight=w_tr)

scores_te = bdt.predict_proba(X_te)[:, 1]
scores_tr = bdt.predict_proba(X_tr)[:, 1]
fpr, tpr, _ = roc_curve(y_te, scores_te,
    sample_weight=w_te)
roc_auc = auc(fpr, tpr)
print(f"\nBDT AUC = {roc_auc:.3f}")

plt.figure(figsize=(6, 5))
plt.plot(tpr, 1 - fpr,
    label=f"BDT (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [1, 0], "k--", label="Random")
plt.xlabel("Signal efficiency")
plt.ylabel("Background rejection")
plt.title(CH_LABEL)
plt.legend(); plt.tight_layout()
plt.savefig(f"figures/roc_{CHANNEL}.png", dpi=200)
plt.show(); plt.close()

plt.figure(figsize=(6, 5))
for lbl, name in [(1, "Signal"), (0, "Background")]:
    plt.hist(scores_tr[y_tr == lbl], bins=40,
        density=True, histtype="step",
        label=f"{name} (train)")
    plt.hist(scores_te[y_te == lbl], bins=40,
        density=True, alpha=0.4,
        label=f"{name} (test)")
plt.xlabel("BDT output")
plt.ylabel("Normalized entries")
plt.title(CH_LABEL)
plt.legend(fontsize=8); plt.tight_layout()
plt.savefig(f"figures/bdt_output_{CHANNEL}.png", dpi=200)
plt.show(); plt.close()

# ------------ Significance scan ------------
best_z, best_cut = 0.0, 0.0
for c in np.linspace(0.0, 1.0, 101):
    s_ = w_te[(y_te == 1) & (scores_te > c)].sum()
    b_ = w_te[(y_te == 0) & (scores_te > c)].sum()
    z  = (s_ / np.sqrt(s_ + b_)
        if (s_ + b_) > 0 else 0.0)
    if z > best_z:
        best_z, best_cut = z, c
print(f"Optimal BDT cut: {best_cut:.2f} "
    f" (Z = {best_z:.1f})")

# ------------ Export figures ------------
shutil.make_archive("figures", "zip", ".", "figures")
try:
    from google.colab import files
    files.download("figures.zip")
except ImportError:
    print("figures.zip saved "
        "(not on Colab, download manually)")
