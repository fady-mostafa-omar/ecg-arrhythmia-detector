"""ECG Arrhythmia Detector — Gradio app (CPU only; runs in Colab or on Hugging Face Spaces)."""
import inspect, os, sys, tempfile
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import gradio as gr
import wfdb

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(HERE, "..", "src"), os.path.join(HERE, "src")):
    if os.path.isdir(_p):
        sys.path.insert(0, os.path.abspath(_p))
from ecgsuite.beats import FS, AAMI, CLASSES, DS2, load_record, bandpass
from ecgsuite.pipeline import analyze, match_to_expert
from ecgsuite.cnn import load_checkpoint as load_model

COLORS = ["#2e9e5b", "#f39c12", "#2563eb", "#8e44ad", "#7f8c8d"]   # V in blue: visible on red ECG paper
NAMES = {"N": "Normal", "S": "Supraventricular ectopic", "V": "Ventricular ectopic",
         "F": "Fusion", "Q": "Unclassifiable"}
WIN, UNSURE = 10, 0.6
GRID_MAJOR, GRID_MINOR, PAPER = "rgba(220,60,60,0.55)", "rgba(220,60,60,0.18)", "#fff6f6"

ICON = {
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "pulse": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
    "heart": '<path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/>',
    "check": '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>',
    "alert": '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
    "bars": '<line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>',
    "repeat": '<polyline points="17 1 21 5 17 9"/><path d="M3 11V9a4 4 0 0 1 4-4h14"/><polyline points="7 23 3 19 7 15"/><path d="M21 13v2a4 4 0 0 1-4 4H3"/>',
    "pause": '<rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/>',
    "zap": '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
    "help": '<circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
    "usercheck": '<path d="M16 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="8.5" cy="7" r="4"/><polyline points="17 11 19 13 23 9"/>',
}

CSS = """
.title {text-align:center; font-size:2.1rem; font-weight:800; margin:8px 0 12px; letter-spacing:-0.02em;}
.sec {font-weight:700; font-size:1.05rem; margin:20px 0 8px; padding-bottom:6px;
      border-bottom:2px solid var(--border-color-primary);}
.kpi-row {display:flex; gap:12px; flex-wrap:wrap; margin:6px 0 12px;}
.kpi {flex:1 1 170px; display:flex; gap:12px; align-items:center; border:1px solid var(--border-color-primary);
      border-radius:14px; padding:12px 14px; background:var(--background-fill-secondary);}
.kpi .ic {flex:0 0 42px; height:42px; border-radius:12px; display:flex; align-items:center; justify-content:center;
          background:rgba(225,29,72,.08); color:#e11d48;}
.kpi.ok .ic {background:rgba(46,158,91,.12); color:#2e9e5b;}
.kpi.warn .ic {background:rgba(243,156,18,.15); color:#d68910;}
.kpi .ic svg {width:22px; height:22px; stroke:currentColor; fill:none; stroke-width:2;
              stroke-linecap:round; stroke-linejoin:round;}
.kpi .v {font-size:1.4rem; font-weight:700; line-height:1.15;}
.kpi .l {font-size:.8rem; opacity:.8;}
.kpi .s {font-size:.72rem; opacity:.6;}
.req {display:flex; flex-wrap:wrap; gap:6px; margin:4px 0 12px;}
.req span {padding:4px 10px; border-radius:999px; font-size:.82rem;
           border:1px solid var(--border-color-primary); background:var(--background-fill-secondary);}
.step {font-weight:700; font-size:1rem; margin:4px 0 2px;}
.pipe {display:flex; flex-wrap:wrap; gap:6px; align-items:center; margin:8px 0 16px;}
.chip {padding:6px 12px; border-radius:999px; border:1px solid var(--border-color-primary);
       background:var(--background-fill-secondary); font-size:.9rem;}
.arrow {opacity:.5;}
table td, table th {font-family: var(--font) !important;}
"""
DESC = ("Finds every heartbeat in a single-lead ECG and flags abnormal beats. "
        "Evaluated on patients the model never saw.")
REQ_HTML = ('<div class="req"><span>Single-lead ECG</span><span>Lead II recommended</span>'
            '<span>CSV · millivolts or raw ADC values</span><span>Any sampling rate ≥ 100 Hz</span>'
            '<span>At least 2 minutes</span><span>Research tool · not a medical device</span></div>')
PIPE_HTML = ('<div class="pipe">' + '<span class="arrow">→</span>'.join(
    f'<span class="chip">{s}</span>' for s in
    ["Raw ECG", "Auto-scale & polarity check", "Band-pass 0.5–40 Hz", "R-peak detection",
     "Beat window + RR intervals", "5 × 1D-CNN ensemble (92k parameters)", "Class + confidence"]) + "</div>")
INPUT_MD = """
### Which recordings work?
| Source | Works? |
|---|---|
| 12-lead hospital ECG | ✅ Choose the **lead II** column |
| AD8232 or another 3-electrode, single-lead device | ⚠️ Usually works; accuracy depends on electrode placement (lead II-like placement is best) |
| Smartwatch or KardiaMobile (lead I) | ⚠️ Untested; accuracy unknown |

**Automatic adjustments for uploads:** if the values do not look like millivolts (for example raw ADC counts
from an AD8232), the signal is rescaled; if the R waves point downward (reversed electrodes), the signal is
inverted. Any adjustment is reported in the Advanced tab.
"""


def sec(title):
    return f'<div class="sec">{title}</div>'


def mmss(s):
    return f"{int(s // 60):02d}:{s % 60:04.1f}"


def cards_html(items):
    out = []
    for icon, label, value, sub, status in items:
        out.append(f'<div class="kpi {status}"><div class="ic"><svg viewBox="0 0 24 24">{ICON[icon]}</svg></div>'
                   f'<div><div class="v">{value}</div><div class="l">{label}</div><div class="s">{sub}</div></div></div>')
    return '<div class="kpi-row">' + "".join(out) + "</div>"


def read_columns(path):
    head = pd.read_csv(path, nrows=5, header=None)
    if pd.to_numeric(head.iloc[0], errors="coerce").isna().all():
        return [str(v) for v in head.iloc[0]], True
    return [f"Column {i + 1}" for i in range(head.shape[1])], False


def default_column(names):
    for n in names:
        if n.strip().upper().replace("_", " ") in ("II", "MLII", "LEAD II", "ECG II"):
            return n
    return names[-1]


def read_signal(path, column):
    names, has_header = read_columns(path)
    df = pd.read_csv(path, header=0 if has_header else None)
    idx = names.index(column) if column in names else len(names) - 1
    return pd.to_numeric(df.iloc[:, idx], errors="coerce").dropna().to_numpy(float)


def auto_adjust(sig, fs):
    notes = []
    f = bandpass(sig, fs=fs)
    hi, lo = np.percentile(f, 99.5), np.percentile(f, 0.5)
    if -lo > 1.5 * hi:
        sig, f = -sig, -f
        notes.append("signal inverted (R waves pointed down)")
    amp = np.percentile(np.abs(f), 99.5)
    if amp > 5 or amp < 0.3:
        k = 1.2 / amp
        sig = sig * k
        notes.append(f"amplitude rescaled ×{k:.3g} (values did not look like millivolts)")
    return sig, notes


def describe_records(data_dir):
    out = []
    for r in [r for r in DS2 if os.path.exists(os.path.join(data_dir, r + ".hea"))]:
        sym = wfdb.rdann(os.path.join(data_dir, r), "atr").symbol
        c = np.bincount([AAMI[s] for s in sym if s in AAMI], minlength=5)
        n = int(c.sum())
        tags = [t for t, ok in (("many ventricular extra beats", c[2] / n > 0.10),
                                ("many supraventricular extra beats", c[1] / n > 0.05)) if ok]
        out.append((f"Patient {r} · {n:,} beats · {', '.join(tags) or 'mostly normal'}", r))
    return out


def vitals(res):
    lab, rr, conf = res["labels"], res["rr"], res["conf"]
    minutes = len(res["signal"]) / FS / 60
    hours = minutes / 60
    n = len(lab)
    sm = pd.Series(60 / rr).rolling(10, center=True, min_periods=5).median().to_numpy()
    sm = sm[(sm > 20) & (sm < 250)]
    off = 100 * np.mean(np.abs(np.diff(rr)) / rr[:-1] > 0.15) if n > 2 else 0.0
    nn = rr[1:][(lab[1:] == 0) & (lab[:-1] == 0)] * 1000
    nn = nn[(nn > 300) & (nn < 2000)]
    return dict(minutes=minutes, n=n, hr=n / minutes,
                hr_min=float(np.min(sm)) if len(sm) else float("nan"),
                hr_max=float(np.max(sm)) if len(sm) else float("nan"),
                off=off, rhythm="Regular" if off < 10 else "Irregular",
                pauses=int((rr > 2.0).sum()),
                n_normal=int((lab == 0).sum()), pct_normal=100 * np.mean(lab == 0),
                n_v=int((lab == 2).sum()), n_s=int((lab == 1).sum()),
                v_h=(lab == 2).sum() / hours, s_h=(lab == 1).sum() / hours,
                pct_unsure=100 * np.mean(conf < UNSURE),
                sdnn=float(np.std(nn)) if len(nn) > 10 else float("nan"),
                rmssd=float(np.sqrt(np.mean(np.diff(nn) ** 2))) if len(nn) > 10 else float("nan"))


def ecg_paper(fig, a_s, b_s, ylim):
    """Style a figure like printed ECG paper: 0.04 s x 0.1 mV small boxes, 0.2 s x 0.5 mV large boxes."""
    xt = np.arange(np.floor(a_s / 0.2) * 0.2, b_s + 1e-9, 0.2)
    yt = np.arange(-ylim, ylim + 1e-9, 0.5)
    lab = lambda v: f"{v:.0f}" if abs(v - round(v)) < 1e-6 else ""
    fig.update_xaxes(range=[a_s, b_s], tickmode="array", tickvals=xt, ticktext=[lab(v) for v in xt],
                     showgrid=True, gridcolor=GRID_MAJOR, gridwidth=1, zeroline=False, ticks="",
                     minor=dict(tickmode="linear", tick0=0, dtick=0.04, showgrid=True,
                                gridcolor=GRID_MINOR, gridwidth=0.5),
                     title_text="time (s)")
    fig.update_yaxes(range=[-ylim, ylim], tickmode="array", tickvals=yt, ticktext=[lab(v) for v in yt],
                     showgrid=True, gridcolor=GRID_MAJOR, gridwidth=1, zeroline=False, ticks="",
                     minor=dict(tickmode="linear", tick0=0, dtick=0.1, showgrid=True,
                                gridcolor=GRID_MINOR, gridwidth=0.5),
                     title_text="mV")
    fig.update_layout(plot_bgcolor=PAPER, paper_bgcolor="white", height=360,
                      margin=dict(l=55, r=20, t=64, b=45), font=dict(family="Inter, sans-serif", size=12),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"))
    return fig


def build_demo(data_dir, model_path, about_md, example_csv=None):
    net, std = load_model(model_path)
    choices = describe_records(data_dir)

    # ---------------- analysis ----------------
    def run_analysis(source, record, upload, column, fs):
        expert, notes = None, []
        if source == "Upload my ECG":
            if not upload:
                raise gr.Error("Upload a CSV file first, or switch to 'Example patient'.")
            fs_in = float(fs or 0)
            if fs_in < 100:
                raise gr.Error("Please enter the sampling rate of your file (at least 100 Hz).")
            sig = read_signal(upload, column)
            if len(sig) < 10 * fs_in:
                raise gr.Error("The recording is shorter than 10 seconds.")
            if len(sig) < 120 * fs_in:
                gr.Warning("Shorter than 2 minutes: heart-rate statistics will be less reliable.")
            sig, notes = auto_adjust(sig, fs_in)
            title = f"Your upload · {len(sig) / fs_in / 60:.1f} min at {fs_in:g} Hz · column '{column}'"
        else:
            sig, samples, symbols = load_record(data_dir, record)
            fs_in = FS
            m = np.isin(symbols, list(AAMI))
            expert = (samples[m], np.array([AAMI[c] for c in symbols[m]]))
            title = f"Patient {record} · MIT-BIH · 30 min · lead MLII · never seen during training"
        res = analyze(sig, fs_in, net, std)
        if len(res["labels"]) < 5:
            raise gr.Error("Fewer than 5 heartbeats were found. Check the column, sampling rate and signal quality.")
        ylim = float(np.clip(np.ceil(np.percentile(np.abs(res["signal"]), 99.9) / 0.5) * 0.5 + 0.5, 1.5, 4.0))
        res.update(conf=res["probs"].max(1), title=title, notes=notes, agree=None, detail="", t0=0.0, ylim=ylim)
        if expert is not None:
            nearest, ok = match_to_expert(res["peaks"], expert[0], int(0.15 * FS))
            yt, yp = expert[1][nearest[ok]], res["labels"][ok]
            res["agree"] = 100 * np.mean(yt == yp)
            res["detail"] = " · ".join(
                f"{NAMES[c]} beats found: {100 * np.mean(yp[yt == k] == k):.0f}% (n={int((yt == k).sum()):,})"
                for k, c in enumerate(CLASSES[:3]) if (yt == k).sum() >= 10)
        return res

    # ---------------- figures ----------------
    def window(res):
        x = res["signal"]
        a = int(res["t0"] * FS)
        b = min(a + WIN * FS, len(x))
        return x, a, b

    def basic_fig(res):
        x, a, b = window(res)
        fig = go.Figure(go.Scatter(x=np.arange(a, b) / FS, y=x[a:b], mode="lines",
                                   line=dict(color="#111", width=1.3), hoverinfo="skip", showlegend=False))
        return ecg_paper(fig, a / FS, a / FS + WIN, res["ylim"])

    def adv_fig(res):
        x, a, b = window(res)
        p, lab, conf, pr = res["peaks"], res["labels"], res["conf"], 100 * res["probs"]
        fig = go.Figure(go.Scatter(x=np.arange(a, b) / FS, y=x[a:b], mode="lines",
                                   line=dict(color="#111", width=1.3), hoverinfo="skip", showlegend=False))
        inw = (p >= a) & (p < b)
        hov = ("<b>%{meta}</b><br>%{text}<br>N %{customdata[0]:.0f}% · S %{customdata[1]:.0f}% · "
               "V %{customdata[2]:.0f}% · F %{customdata[3]:.0f}% · Q %{customdata[4]:.0f}%<extra></extra>")
        for k, c in enumerate(CLASSES):
            for unsure in (False, True):
                sel = inw & (lab == k) & ((conf < UNSURE) == unsure)
                if not sel.any():
                    continue
                ts = p[sel] / FS
                fig.add_trace(go.Scatter(
                    x=ts, y=x[p[sel]], mode="markers", legendgroup=c, meta=NAMES[c],
                    name=f"{c} · {NAMES[c]}" + (" (low confidence)" if unsure else ""),
                    marker=dict(size=12, color="white" if unsure else COLORS[k],
                                line=dict(width=2.5 if unsure else 1.5, color=COLORS[k] if unsure else "white")),
                    text=[mmss(t) for t in ts], customdata=pr[sel], hovertemplate=hov))
        return ecg_paper(fig, a / FS, a / FS + WIN, res["ylim"])

    def tacho(res):
        t, rr, lab = res["peaks"] / FS / 60, res["rr"] * 1000, res["labels"]
        fig = go.Figure()
        for k, c in enumerate(CLASSES):
            sel = lab == k
            if sel.any():
                fig.add_trace(go.Scattergl(x=t[sel], y=rr[sel], mode="markers", name=f"{c} · {NAMES[c]}",
                                           marker=dict(size=3 if k == 0 else 5, color=COLORS[k])))
        fig.update_layout(template="plotly_white", height=300, margin=dict(l=55, r=10, t=40, b=45),
                          title=dict(text="Beat-to-beat interval (RR) across the recording", font=dict(size=14)),
                          xaxis_title="time (min)", yaxis_title="RR interval (ms)",
                          font=dict(family="Inter, sans-serif", size=12),
                          legend=dict(orientation="h", yanchor="bottom", y=1.0, x=1, xanchor="right"))
        return fig

    def bar(res):
        n = np.bincount(res["labels"], minlength=5)
        fig = go.Figure(go.Bar(x=n, y=[f"{c} · {NAMES[c]}" for c in CLASSES], orientation="h",
                               marker_color=COLORS, text=[f"{v:,}" for v in n], textposition="auto"))
        fig.update_layout(template="plotly_white", height=300, margin=dict(l=10, r=10, t=40, b=40),
                          title=dict(text="Beats per class", font=dict(size=14)),
                          yaxis=dict(autorange="reversed"), font=dict(family="Inter, sans-serif", size=12))
        return fig

    # ---------------- text, cards, tables ----------------
    def summary_md(res, v):
        hr = v["hr"]
        where = ("below" if hr < 60 else "above" if hr > 100 else "within") + \
                " the usual resting range for adults (60–100)"
        lines = [f"**{res['title']}**",
                 f"Over **{v['minutes']:.1f} minutes** the heart beat **{v['n']:,} times**, about "
                 f"**{hr:.0f} beats per minute** on average, {where}.",
                 f"**{v['pct_normal']:.0f}%** of the beats looked normal."]
        if v["n_v"]:
            lines.append(f"**{v['n_v']:,} extra beats from the lower chambers** (ventricular), about "
                         f"**{v['v_h']:.0f} per hour**.")
        if v["n_s"]:
            lines.append(f"**{v['n_s']:,} extra beats from the upper chambers** (supraventricular), about "
                         f"**{v['s_h']:.0f} per hour**.")
        lines.append(f"The rhythm looks **{v['rhythm'].lower()}** ({v['off']:.0f}% of beats came noticeably "
                     f"early or late). " + (f"**{v['pauses']} pause(s)** longer than 2 seconds."
                                            if v["pauses"] else "No pauses longer than 2 seconds."))
        if res["notes"]:
            lines.append("*Automatic adjustment: " + "; ".join(res["notes"]) + ".*")
        return "\n\n".join(lines)

    def st(ok):
        return "ok" if ok else "warn"

    def basic_cards(v):
        return cards_html([
            ("clock", "Recording length", f"{v['minutes']:.1f} min", "", ""),
            ("pulse", "Heartbeats", f"{v['n']:,}", "detected", ""),
            ("heart", "Average heart rate", f"{v['hr']:.0f} bpm", "usual at rest: 60–100", st(60 <= v["hr"] <= 100)),
            ("check", "Normal beats", f"{v['pct_normal']:.1f}%", f"{v['n_normal']:,} beats", "ok"),
            ("alert", "Abnormal beats", f"{100 - v['pct_normal']:.1f}%", f"{v['n'] - v['n_normal']:,} beats", ""),
        ])

    def hr_cards(v):
        return cards_html([
            ("heart", "Average heart rate", f"{v['hr']:.0f} bpm", "usual at rest: 60–100", st(60 <= v["hr"] <= 100)),
            ("bars", "Lowest / highest rate", f"{v['hr_min']:.0f} / {v['hr_max']:.0f} bpm", "10-beat average", ""),
            ("repeat", "Rhythm", v["rhythm"], f"{v['off']:.0f}% of beats off-timing", st(v["rhythm"] == "Regular")),
            ("pause", "Pauses longer than 2 s", f"{v['pauses']}", "", st(v["pauses"] == 0)),
        ])

    def cls_cards(v):
        return cards_html([
            ("check", "Normal", f"{v['pct_normal']:.1f}%", f"{v['n_normal']:,} beats", "ok"),
            ("alert", "Abnormal (S, V, F, Q)", f"{100 - v['pct_normal']:.1f}%", f"{v['n'] - v['n_normal']:,} beats", ""),
            ("zap", "Extra beats · lower chambers (V)", f"{v['v_h']:.0f} / hour", f"{v['n_v']:,} in total", st(v["v_h"] <= 30)),
            ("zap", "Extra beats · upper chambers (S)", f"{v['s_h']:.0f} / hour", f"{v['n_s']:,} in total", st(v["s_h"] <= 30)),
        ])

    def conf_cards(res, v):
        items = [("help", "Low-confidence beats", f"{v['pct_unsure']:.1f}%", "confidence below 60% (hollow markers)", "")]
        if res["agree"] is not None:
            items.append(("usercheck", "Agreement with cardiologists", f"{res['agree']:.1f}%", "beat by beat", ""))
        return cards_html(items)

    def conf_md(res):
        if res["detail"]:
            return "**Per class, compared with the cardiologists' labels:** " + res["detail"]
        return "*Agreement with cardiologists is available for the example patients only.*"

    def hrv_cards(v):
        return cards_html([
            ("bars", "SDNN", f"{v['sdnn']:.0f} ms", "variability of normal-to-normal intervals", ""),
            ("pulse", "RMSSD", f"{v['rmssd']:.0f} ms", "beat-to-beat variability", ""),
        ])

    def table(res):
        lab, conf = res["labels"], res["conf"]
        return pd.DataFrame({"Class": [f"{c} · {NAMES[c]}" for c in CLASSES],
                             "Beats": [int((lab == k).sum()) for k in range(5)],
                             "% of all beats": [round(100 * np.mean(lab == k), 1) for k in range(5)],
                             "Low confidence": [int(((lab == k) & (conf < UNSURE)).sum()) for k in range(5)]})

    def export(res):
        df = pd.DataFrame({"time_s": (res["peaks"] / FS).round(3), "rr_ms": (res["rr"] * 1000).round(1),
                           "class": [CLASSES[k] for k in res["labels"]], "confidence": res["conf"].round(3)})
        for k, c in enumerate(CLASSES):
            df[f"p_{c}"] = res["probs"][:, k].round(3)
        f = tempfile.NamedTemporaryFile(delete=False, suffix="_beats.csv")
        df.to_csv(f.name, index=False)
        return f.name

    # ---------------- navigation ----------------
    def position(res):
        dur = len(res["signal"]) / FS
        return f"<center>Showing <b>{mmss(res['t0'])} – {mmss(min(res['t0'] + WIN, dur))}</b> of {mmss(dur)}</center>"

    def views(res):
        return basic_fig(res), position(res), adv_fig(res), position(res), res

    def shift(res, step):
        if res is None:
            raise gr.Error("Analyze an ECG first.")
        dur = len(res["signal"]) / FS
        res["t0"] = float(np.clip(res["t0"] + step, 0, max(dur - WIN, 0)))
        return views(res)

    def go_prev(res):
        return shift(res, -WIN)

    def go_next(res):
        return shift(res, WIN)

    def next_abnormal(res):
        if res is None:
            raise gr.Error("Analyze an ECG first.")
        t, ab = res["peaks"] / FS, res["labels"] != 0
        idx = np.where(ab & (t > res["t0"] + WIN / 2 + 0.5))[0]
        if len(idx) == 0:
            idx = np.where(ab)[0]
        if len(idx) == 0:
            gr.Info("No abnormal beats in this recording.")
            return views(res)
        dur = len(res["signal"]) / FS
        res["t0"] = float(np.clip(t[idx[0]] - WIN / 2, 0, max(dur - WIN, 0)))
        return views(res)

    def render(res):
        v = vitals(res)
        return (basic_cards(v), basic_fig(res), position(res),
                summary_md(res, v), adv_fig(res), position(res),
                hr_cards(v), tacho(res), cls_cards(v), bar(res), table(res),
                conf_cards(res, v), conf_md(res), hrv_cards(v), export(res), res)

    def on_upload(f):
        if not f:
            return gr.update(choices=[], value=None)
        names, _ = read_columns(f)
        return gr.update(choices=names, value=default_column(names))

    # ---------------- layout ----------------
    theme = gr.themes.Soft(primary_hue="rose", neutral_hue="slate",
                           font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
                           font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"])
    if "theme" in inspect.signature(gr.Blocks.__init__).parameters:
        blocks_kw, launch_kw = dict(theme=theme, css=CSS), {}
    else:
        blocks_kw, launch_kw = {}, dict(theme=theme, css=CSS)

    with gr.Blocks(title="ECG Arrhythmia Detector", **blocks_kw) as demo:
        state = gr.State()
        gr.HTML('<div class="title">ECG Arrhythmia Detector</div>')
        with gr.Group():
            with gr.Row(equal_height=False):
                with gr.Column(scale=2):
                    gr.HTML('<div class="step">1 · Choose an ECG</div>')
                    source = gr.Radio(["Example patient", "Upload my ECG"], value="Example patient",
                                      show_label=False)
                    record = gr.Dropdown(choices, value="233", label="Example patient (not used for training)")
                    with gr.Row(visible=False) as up_row:
                        upload = gr.File(label="Your ECG file (.csv)", file_types=[".csv", ".txt"],
                                         type="filepath", scale=2)
                        with gr.Column(scale=1):
                            column = gr.Dropdown([], label="ECG column", allow_custom_value=True)
                            fs = gr.Number(360, label="Sampling rate (Hz)", precision=0)
                with gr.Column(scale=1):
                    gr.HTML('<div class="step">2 · Analyze</div>')
                    run_btn = gr.Button("Analyze ECG", variant="primary", size="lg")
                    if example_csv:
                        gr.File(value=example_csv, label="Example file to try the upload (5 min, 360 Hz)",
                                interactive=False)
        with gr.Tabs():
            with gr.Tab("Basic"):
                b_cards = gr.HTML()
                b_fig = gr.Plot(show_label=False)
                with gr.Row():
                    b_prev = gr.Button("◀ 10 s", size="sm")
                    b_pos = gr.Markdown()
                    b_next = gr.Button("10 s ▶", size="sm")
            with gr.Tab("Advanced"):
                gr.HTML(sec("Recording"))
                a_summary = gr.Markdown()
                gr.HTML(sec("Signal viewer"))
                a_fig = gr.Plot(show_label=False)
                with gr.Row():
                    a_prev = gr.Button("◀ 10 s", size="sm")
                    a_pos = gr.Markdown()
                    a_next = gr.Button("10 s ▶", size="sm")
                    a_abn = gr.Button("Next abnormal beat ⏭", size="sm", variant="secondary")
                gr.HTML(sec("Heart rate & rhythm"))
                a_hr = gr.HTML()
                t_fig = gr.Plot(show_label=False)
                gr.HTML(sec("Beat classification"))
                a_cls = gr.HTML()
                with gr.Row():
                    c_fig = gr.Plot(show_label=False)
                    tbl = gr.Dataframe(interactive=False, label="Beats per class")
                gr.HTML(sec("Model confidence & validation"))
                a_conf = gr.HTML()
                a_conf_md = gr.Markdown()
                gr.HTML(sec("Heart rate variability"))
                a_hrv = gr.HTML()
                gr.HTML(sec("Export"))
                csv = gr.File(label="Beat-by-beat results (CSV)")
            with gr.Tab("How it works"):
                gr.Markdown(DESC)
                gr.HTML(REQ_HTML)
                gr.HTML(PIPE_HTML)
                gr.Markdown(INPUT_MD)
                gr.Markdown(about_md)
        gr.Markdown("<center><small>Fady Mostafa Omar · 2026 · Data: MIT-BIH Arrhythmia Database "
                    "(PhysioNet)</small></center>")

        outs = [b_cards, b_fig, b_pos, a_summary, a_fig, a_pos, a_hr, t_fig, a_cls, c_fig, tbl,
                a_conf, a_conf_md, a_hrv, csv, state]
        nav = [b_fig, b_pos, a_fig, a_pos, state]
        source.change(lambda s: (gr.update(visible=s == "Example patient"),
                                 gr.update(visible=s != "Example patient")), source, [record, up_row])
        upload.change(on_upload, upload, column)
        run_btn.click(lambda s, r, u, c, f: render(run_analysis(s, r, u, c, f)),
                      [source, record, upload, column, fs], outs)
        record.change(lambda r: render(run_analysis("Example patient", r, None, None, FS)), record, outs)
        demo.load(lambda: render(run_analysis("Example patient", "233", None, None, FS)), None, outs)
        for btn, fn in ((b_prev, go_prev), (b_next, go_next), (a_prev, go_prev), (a_next, go_next),
                        (a_abn, next_abnormal)):
            btn.click(fn, state, nav)
    return demo, launch_kw



def build_about(results_dir):
    """Text of the 'How it works' tab, filled from the files written by scripts/evaluate.py."""
    import json
    e2e = pd.read_csv(os.path.join(results_dir, "end_to_end_ds2.csv"), index_col=0)
    rp = pd.read_csv(os.path.join(results_dir, "rpeak_ds2.csv"), index_col=0)
    s_rec = pd.read_csv(os.path.join(results_dir, "s_per_record.csv")).sort_values("S beats", ascending=False)
    with open(os.path.join(results_dir, "model_info.json")) as f:
        info = json.load(f)
    tp = (rp["expert beats"] - rp["missed"]).sum()
    det_se, det_ppv = 100 * tp / rp["expert beats"].sum(), 100 * tp / (tp + rp["extra"].sum())
    rows = "\n".join(f"| {c} | {int(e2e.loc[c, 'n_test']):,} | {e2e.loc[c, 'Se %']:.1f} | {e2e.loc[c, 'PPV %']:.1f} |"
                     for c in ["N", "S", "V", "F"])
    top = s_rec.iloc[0]
    f_line = ("Fusion (F) beats are not recognized." if e2e.loc["F", "Se %"] < 5 else
              f"Fusion (F) beats are rarely recognized (sensitivity {e2e.loc['F', 'Se %']:.0f}%).")
    return f"""
### What the five classes mean
| Class | Meaning |
|---|---|
| **N** | Normal beat (including bundle branch block beats) |
| **S** | Supraventricular ectopic: an early beat starting above the ventricles |
| **V** | Ventricular ectopic: an early beat starting in the ventricles (wide, abnormal shape) |
| **F** | Fusion of a normal and a ventricular beat |
| **Q** | Unclassifiable |

### How it was tested
Trained only on the 22 MIT-BIH patients of DS1 (17 to train, 5 to decide when to stop) and tested once on
22 **different** patients (DS2), the inter-patient protocol of de Chazal et al. (2004).

- **Heartbeat detection:** sensitivity {det_se:.2f}%, positive predictivity {det_ppv:.2f}% (150 ms window)
- **Model:** {info['members']} compact 1D-CNNs trained with different random seeds; their probabilities are averaged
- **Size and speed:** {info['parameters']:,} parameters; about {info['us_per_beat_cpu'] / 1000:.1f} ms per beat on a CPU

**Beat classification, end to end on DS2** (raw ECG → detector → model, compared with the cardiologists' labels):

| Class | Test beats | Sensitivity % | Positive predictivity % |
|---|---|---|---|
{rows}

### Limitations
- Trained on MIT-BIH only (47 people, one recorder type, lead MLII); other devices and leads are untested.
- About {info['normal_mislabeled_pct']:.0f} in 100 normal beats are labeled abnormal, so the abnormal share is overestimated in mostly normal recordings.
- Supraventricular (S) beats are often missed; one test patient (record {int(top['record'])}) holds {top['share of all S %']:.0f}% of all DS2 S beats, so the S result mostly reflects that patient.
- {f_line}
- Heart-rate statistics and pauses depend on beat detection; missed beats can create false pauses.
- Research and education tool, not a medical device.

### Data and citation
MIT-BIH Arrhythmia Database (Moody & Mark, 2001), PhysioNet, Open Data Commons Attribution License.
2026 re-implementation using public data only.
"""


def main():
    import argparse
    root = os.path.join(HERE, "..")
    p = argparse.ArgumentParser(description="ECG Arrhythmia Detector (Gradio app)")
    p.add_argument("--data", default=os.path.join(root, "data", "mitdb"), help="folder with the MIT-BIH records")
    p.add_argument("--model", default=os.path.join(root, "models", "beatnet.pt"))
    p.add_argument("--results", default=os.path.join(root, "results"))
    p.add_argument("--share", action="store_true", help="create a temporary public gradio.live link")
    a = p.parse_args()
    example = os.path.join(HERE, "examples", "example_ecg_360Hz.csv")
    demo, launch_kw = build_demo(a.data, a.model, build_about(a.results),
                                 example if os.path.exists(example) else None)
    demo.launch(share=a.share, **launch_kw)


if __name__ == "__main__":
    main()
