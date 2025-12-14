import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import uuid
import pickle
import re
import zipfile
import io
from pathlib import Path

# ---------------------------------------------------------
# Project imports
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from flowgraph.tx_rx_simulation import tx_rx_simulation


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def safe_name(s: str) -> str:
    """Make a filesystem-safe name."""
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-zA-Z0-9_\-\.]+", "", s)
    return s or "item"


# ---------------------------------------------------------
# Session state init
# ---------------------------------------------------------
if "phase" not in st.session_state:
    st.session_state.phase = "setup"

if "experiment_params" not in st.session_state:
    st.session_state.experiment_params = {}

if "curves" not in st.session_state:
    st.session_state.curves = []

if "show_final_plot" not in st.session_state:
    st.session_state.show_final_plot = False


# ---------------------------------------------------------
# Simulation core
# ---------------------------------------------------------
def run_curve(snrs, sf, cr, soft_decoding,
              n_frames, samp_rate, bw,
              center_freq, clk_offset_ppm,
              pay_len, ldro):

    os.makedirs("data/temp", exist_ok=True)

    with open("data/payload_master1.txt") as f:
        master_payloads = [p.strip() for p in f.readlines()]

    FER, Glob_FER = [], []
    run_id = uuid.uuid4().hex[:8]

    for snr_idx, SNRdB in enumerate(snrs):
        tx = f"data/temp/tx_{run_id}_{snr_idx}.txt"
        rx = f"data/temp/rx_{run_id}_{snr_idx}.txt"
        crc = f"data/temp/crc_{run_id}_{snr_idx}.bin"

        with open(tx, "w") as ftx:
            for p in master_payloads[:n_frames]:
                ftx.write(p + ",")

        tb = tx_rx_simulation(
            tx, rx, crc,
            impl_head=False,
            soft_decoding=soft_decoding,
            SNRdB=float(SNRdB),
            samp_rate=samp_rate,
            bw=bw,
            center_freq=center_freq,
            sf=sf,
            cr=cr,
            pay_len=pay_len,
            clk_offset_ppm=clk_offset_ppm,
            ldro=ldro
        )

        tb.start()
        tb.wait()

        if not os.path.exists(crc):
            FER.append(np.nan)
            Glob_FER.append(np.nan)
            continue

        rx_crc = np.fromfile(crc, dtype=np.uint8)
        if len(rx_crc) == 0:
            FER.append(np.nan)
            Glob_FER.append(np.nan)
        else:
            FER.append(1 - np.sum(rx_crc) / len(rx_crc))
            Glob_FER.append(1 - np.sum(rx_crc) / n_frames)

    return np.array(FER), np.array(Glob_FER)


# ---------------------------------------------------------
# FINAL PLOT (exclusive view) — shows ONLY the final plot
# ---------------------------------------------------------
if st.session_state.show_final_plot:
    st.header("Final comparison (all saved curves)")

    fig, ax = plt.subplots()
    for c in st.session_state.curves:
        ax.semilogy(c["snrs"], c["FER"], "-d", label=c["label"])

    ax.set_xlabel("SNR [dB]")
    ax.set_ylabel("FER")
    ax.grid(True, which="both")
    ax.legend()

    st.pyplot(fig)

    # Save final plot image (only when user opens this view)
    os.makedirs("results/final", exist_ok=True)
    final_plot_path = "results/final/final_plot.png"
    fig.savefig(final_plot_path, dpi=300, bbox_inches="tight")

    # Download final plot
    with open(final_plot_path, "rb") as f:
        st.download_button(
            "Download final plot (PNG)",
            data=f,
            file_name="final_plot.png",
            mime="image/png",
            key="dl_final_plot_png"
        )

    if st.button("Hide final plot", key="hide_final_plot_btn"):
        st.session_state.show_final_plot = False
        st.rerun()

    st.stop()


# ---------------------------------------------------------
# SETUP PHASE
# ---------------------------------------------------------
if st.session_state.phase == "setup":
    st.header("Experiment setup")

    sf = st.number_input("Spreading Factor (SF)", value=7, step=1)
    snr_min = st.number_input("SNR min [dB]", value=-13.0)
    snr_max = st.number_input("SNR max [dB]", value=-5.0)
    snr_step = st.number_input("SNR step [dB]", value=0.5)
    n_frames = st.number_input("Frames per SNR", value=200, step=50)

    if st.button("Save experiment settings", key="save_settings_btn"):
        st.session_state.experiment_params = {
            "sf": sf,
            "snr_min": snr_min,
            "snr_max": snr_max,
            "snr_step": snr_step,
            "n_frames": int(n_frames),
            "samp_rate": 500_000,
            "bw": 125_000,
            "center_freq": 868.1,
            "clk_offset_ppm": 0,
            "pay_len": 16,
            "ldro": False,
        }

        st.session_state.phase = "run"
        st.success("Experiment settings saved. Now run curves.")
        st.rerun()


# ---------------------------------------------------------
# RUN PHASE
# ---------------------------------------------------------
if st.session_state.phase == "run":
    st.header("Run curves")

    cr = st.selectbox("Code rate (CR)", [1, 2, 3, 4], index=1, key="cr_select")
    soft_decoding = st.checkbox("Soft decoding", value=False, key="soft_chk")

    plot_placeholder = st.empty()
    status_text = st.empty()

    if st.button("Run curve", key="run_curve_btn"):

        duplicate = any(
            c.get("cr") == cr and c.get("soft_decoding") == soft_decoding
            for c in st.session_state.curves
        )

        if duplicate:
            st.warning(
                f"⚠️ Curve with CR={cr} and "
                f"{'soft' if soft_decoding else 'hard'} decoding "
                "has already been simulated."
            )
            st.stop()  # ⛔ ΣΤΑΜΑΤΑΕΙ ΕΔΩ

        p = st.session_state.experiment_params

        snrs = np.arange(
            p["snr_min"],
            p["snr_max"] + p["snr_step"],
            p["snr_step"]
        )

        with st.spinner("Simulating..."):
            FER, Glob_FER = run_curve(
                snrs,
                p["sf"],
                cr,
                soft_decoding,
                p["n_frames"],
                p["samp_rate"],
                p["bw"],
                p["center_freq"],
                p["clk_offset_ppm"],
                p["pay_len"],
                p["ldro"]
            )

        status_text.success("Simulation completed ✅")

        label = f"CR={cr} ({'soft' if soft_decoding else 'hard'})"

        # Create curve folder (unique per run)
        os.makedirs("results/curves", exist_ok=True)
        run_tag = uuid.uuid4().hex[:6]
        curve_dir = f"results/curves/{safe_name(label)}_{run_tag}"
        os.makedirs(curve_dir, exist_ok=True)

        curve_img_path = f"{curve_dir}/curve.png"
        curve_pkl_path = f"{curve_dir}/curve.pkl"

        # Plot current curve
        fig, ax = plt.subplots()
        ax.semilogy(snrs, FER, "-d", label="FER")
        ax.semilogy(snrs, Glob_FER, "-d", label="FER including frame miss")
        ax.set_title(label)
        ax.set_xlabel("SNR [dB]")
        ax.set_ylabel("Error rate")
        ax.grid(True, which="both")
        ax.legend()

        # Show plot in app
        plot_placeholder.pyplot(fig)

        # Save PNG
        fig.savefig(curve_img_path, dpi=300, bbox_inches="tight")

        # Save PKL
        with open(curve_pkl_path, "wb") as f:
            pickle.dump(
                {
                    "snrs": snrs,
                    "FER": FER,
                    "Glob_FER": Glob_FER,
                    "label": label,
                    "params": p,
                    "cr": cr,
                    "soft_decoding": soft_decoding,
                },
                f
            )

        # Store in session
        st.session_state.curves.append({
            "label": label,
            "snrs": snrs,
            "FER": FER,
            "Glob_FER": Glob_FER,
            "cr": cr,
            "soft_decoding": soft_decoding,
            "curve_img": curve_img_path,
            "curve_pkl": curve_pkl_path,
            "curve_dir": curve_dir,
        })

def build_zip_from_curves(curves):
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:

        # --------------------------------------------------
        # 1. Βάλε όλα τα individual curve files
        # --------------------------------------------------
        for i, c in enumerate(curves, start=1):
            # plot
            zf.write(
                c["curve_img"],
                arcname=f"curves/curve_{i}/plot.png"
            )

            # pkl
            zf.write(
                c["curve_pkl"],
                arcname=f"curves/curve_{i}/data.pkl"
            )

        # --------------------------------------------------
        # 2. Δημιουργία FINAL PLOT (όπως στο UI)
        # --------------------------------------------------
        fig, ax = plt.subplots()

        for c in curves:
            ax.semilogy(
                c["snrs"],
                c["FER"],
                "-d",
                label=c["label"]
            )

        ax.set_xlabel("SNR [dB]")
        ax.set_ylabel("FER")
        ax.grid(True, which="both")
        ax.legend()

        # save σε προσωρινό buffer
        img_buf = io.BytesIO()
        fig.savefig(img_buf, dpi=300, bbox_inches="tight")
        plt.close(fig)
        img_buf.seek(0)

        zf.writestr("final/final_plot.png", img_buf.read())

        # --------------------------------------------------
        # 3. FINAL PKL (όλες οι curves μαζί)
        # --------------------------------------------------
        final_data = {
            "curves": curves,
            "n_curves": len(curves),
        }

        pkl_buf = io.BytesIO()
        pickle.dump(final_data, pkl_buf)
        pkl_buf.seek(0)

        zf.writestr("final/final_plot.pkl", pkl_buf.read())

    zip_buffer.seek(0)
    return zip_buffer




# ---------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------
with st.sidebar:
    st.header("Controls")

    has_curves = len(st.session_state.curves) > 0

    # ==================================================
    # FINAL PLOT
    # ==================================================
    if st.button(
        "Show final plot",
        key="show_final_plot_btn",
        disabled=not has_curves
    ):
        st.session_state.show_final_plot = True
        st.rerun()

    # ==================================================
    # RESET
    # ==================================================
    if st.button(
        "Reset experiment",
        key="reset_experiment_btn"
    ):
        st.session_state.phase = "setup"
        st.session_state.experiment_params = {}
        st.session_state.curves = []
        st.session_state.show_final_plot = False
        st.rerun()

    # ==================================================
    # DOWNLOAD RESULTS
    # ==================================================


    if has_curves:
        st.divider()
        st.header("Download results")
        # ---------- ZIP ALL ----------
        zip_data = build_zip_from_curves(st.session_state.curves)

        st.download_button(
            label="⬇ Download ALL results (ZIP)",
            data=zip_data,
            file_name="lora_fer_results.zip",
            mime="application/zip",
            key="download_all_zip"
        )

        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

        # ---------- INDIVIDUAL CURVES ----------
        for i, c in enumerate(st.session_state.curves, start=1):

            # Τίτλος curve (μεγάλος αριθμός αριστερά)
            st.markdown(
                f"""
                <div style="
                    display:flex;
                    align-items:center;
                    gap:10px;
                    margin-bottom:4px;
                ">
                    <div style="
                        font-size:28px;
                        font-weight:800;
                        color:#2E7D32;
                        width:28px;
                        text-align:right;
                    ">
                        {i}
                    </div>
                    <div style="font-size:18px;">
                        <b>{c['label']}</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Κουμπιά plot / data
            col1, col2 = st.columns([1, 1])

            with col1:
                with open(c["curve_img"], "rb") as f:
                    st.download_button(
                        "⬇ Plot",
                        f,
                        file_name=f"curve_{i}.png",
                        mime="image/png",
                        key=f"dl_plot_{i}"
                    )

            with col2:
                with open(c["curve_pkl"], "rb") as f:
                    st.download_button(
                        "⬇ Data",
                        f,
                        file_name=f"curve_{i}.pkl",
                        mime="application/octet-stream",
                        key=f"dl_data_{i}"
                    )

            # μικρό κενό (σφιχτό)
            st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

