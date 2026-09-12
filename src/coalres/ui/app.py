"""UI Streamlit: menjalankan 11 tahap dan menampilkan gerbangnya.

Satu aturan yang membentuk seluruh berkas ini: UI TIDAK BOLEH MELEWATI GERBANG.
Ia menampilkan apa yang menghentikan run dan apa yang harus diputuskan manusia;
ia tidak menyediakan tombol "lanjut saja". Tombol semacam itu akan menjadi jalan
paling mudah menuju angka yang tidak boleh dipakai.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Impor ABSOLUT, bukan relatif: `streamlit run app.py` menjalankan berkas ini
# sebagai skrip tingkat atas, dan impor relatif gagal di sana. Absolut bekerja
# di kedua konteks - dijalankan streamlit maupun diimpor sebagai modul.
from coalres.config import Config
from coalres.errors import ConfigError
from coalres.ui import pipeline

STAGE_NAMES = {
    "2_audit": "Tahap 2 - Audit data",
    "5_validasi": "Tahap 5 - Validasi model",
    "9_batas": "Tahap 9 - Batas pelaporan (KCMI 4.6)",
    "7_kompleksitas": "Tahap 7 - Pembobotan kompleksitas geologi",
    "10_swauji": "Tahap 10 - Swauji peta",
}


def _config_path() -> Path:
    """Konfigurasi awal dari COALRES_CONFIG, disetel peluncur CLI.

    Streamlit menjalankan berkas ini sebagai skrip tingkat atas, jadi argumen
    baris perintah tidak sampai ke sini; peluncur meneruskannya lewat lingkungan.
    """
    import os

    st.sidebar.header("Konfigurasi")
    default = os.environ.get("COALRES_CONFIG", "config/minex_dummy.yaml")
    path = st.sidebar.text_input("Berkas konfigurasi", value=default,
                                 key="cfgpath")
    return Path(path)


def _load(path: Path) -> Config | None:
    try:
        return Config.load(path)
    except ConfigError as exc:
        st.sidebar.error(f"Konfigurasi tidak valid:\n\n{exc}")
        return None


def tab_audit(stages) -> None:
    report = stages.audit
    stops, warns = report.stops, report.of(1)
    a, b, c = st.columns(3)
    a.metric("STOP", len(stops))
    b.metric("WARN", len(warns))
    c.metric("INFO", len(report.of(0)))

    if stops:
        st.error(f"Phase 0 TIDAK LULUS - {len(stops)} gerbang harus "
                 "diselesaikan manusia sebelum estimasi dijalankan.")
        for finding in stops:
            with st.expander(f"STOP  {finding.check}", expanded=True):
                st.write(finding.message)
                if finding.remedy:
                    st.info(f"Yang harus dilakukan: {finding.remedy}")
    else:
        st.success("Phase 0 lulus.")

    with st.expander(f"{len(warns)} peringatan"):
        for finding in warns:
            st.write(f"**{finding.check}** - {finding.message}")
    for name, table in report.tables.items():
        with st.expander(f"Tabel audit: {name}"):
            st.dataframe(table, width="stretch")


def _complexity_form(stages) -> None:
    """Formulir ceklis untuk subaspek yang menunggu keputusan manusia."""
    from coalres import complexity as cx

    st.warning(
        "Pembobotan menunggu keputusan Anda. Program tidak memberi nilai "
        "bawaan: kondisi geologi menggerakkan SELURUH radius klasifikasi.")
    suggestions = {s.parameter: s for s in (stages.suggestions or [])}
    stored = dict(stages.cfg.complexity_input.as_overrides())

    with st.form("kompleksitas"):
        chosen: dict[str, tuple[str, str]] = {}
        for group, params in cx.FORM.items():
            st.markdown(f"**Aspek {group}**")
            for parameter, wording in params.items():
                suggestion = suggestions.get(parameter)
                default = (stored.get(parameter, (None, ""))[0]
                           or (suggestion.score if suggestion and suggestion.assessable
                               else None))
                columns = st.columns([2, 3, 5])
                columns[0].write(parameter)
                score = columns[1].radio(
                    parameter, cx.SCORES, horizontal=False,
                    index=cx.SCORES.index(default) if default else None,
                    label_visibility="collapsed", key=f"skor_{parameter}")
                justification = columns[2].text_area(
                    parameter, value=stored.get(parameter, ("", ""))[1],
                    height=68, label_visibility="collapsed",
                    placeholder=(f"Usulan: {suggestion.evidence}"
                                 if suggestion else
                                 f"Justifikasi (min {cx.JUSTIFICATION_MIN_CHARS} karakter)"),
                    key=f"just_{parameter}")
                if suggestion:
                    st.caption(f"{wording[score] if score else ''} - "
                               f"usulan otomatis: {suggestion.evidence}. "
                               f"Ambang: {suggestion.threshold}")
                if score and justification.strip():
                    chosen[parameter] = (score, justification.strip())
        submitted = st.form_submit_button("Terapkan dan jalankan ulang tahap 7-10",
                                          type="primary")
    if submitted:
        with st.spinner("Menjalankan ulang..."):
            st.session_state["stages"] = pipeline.run(
                stages.cfg, spacing=st.session_state.get("spacing", 25.0),
                complexity_overrides=chosen,
                cross_validation=st.session_state.get("cross", True))
        st.rerun()


def tab_complexity(stages) -> None:
    assessment = stages.assessment
    if assessment is None:
        if stages.stopped_at == "7_kompleksitas":
            _complexity_form(stages)
        else:
            st.info("Tahap 7 belum dijalankan.")
        return
    st.subheader(f"Kondisi geologi: {assessment.condition.upper()}")
    st.caption(
        "Nilai dihitung dua tingkat: subaspek dirata-ratakan menjadi nilai "
        "aspek, lalu ketiga aspek dirata-ratakan. Ceklis, bukan angka.")
    st.dataframe(assessment.to_checklist_frame().fillna(""), width="stretch")
    for warning in assessment.warnings:
        st.warning(warning)
    if stages.radii:
        st.markdown("**Radius yang berlaku (SNI 5015:2019)**")
        st.dataframe(pd.DataFrame([stages.radii]), width="stretch")


def tab_observation(stages) -> None:
    points = stages.observation
    if points is None:
        st.info("Tahap 6 belum dijalankan.")
        return
    st.caption(
        "Titik observasi menuntut ketebalan DAN kualitas, dihitung PER SEAM. "
        "Lubang yang punya kualitas pada satu seam tidak menjadi titik "
        "observasi bagi seam lain di lubang yang sama.")
    st.dataframe(points.summary(), width="stretch")
    if stages.spotted_dog is not None and not stages.spotted_dog.empty:
        failing = stages.spotted_dog[~stages.spotted_dog["memenuhi_kcmi"]]
        st.markdown(f"**KCMI 4.5.4 / 4.5.5** - {len(failing)} dari "
                    f"{len(stages.spotted_dog)} bagian tidak memenuhi")
        st.dataframe(stages.spotted_dog, width="stretch")
    if stages.poo_criteria is not None and stages.poo_criteria.undeclared:
        st.warning("KCMI 4.5.2 belum dinyatakan: "
                   + ", ".join(stages.poo_criteria.undeclared))


def tab_validation(stages) -> None:
    report = stages.validation
    if report is None:
        st.info("Tahap 5 belum dijalankan.")
        return
    if report.failures:
        st.error(f"{len(report.failures)} kegagalan - jalurnya KEMBALI ke tahap 4.")
        for line in report.failures:
            st.write(f"- {line}")
    else:
        st.success("Validasi lulus.")
    st.caption(
        "'menghormati' membaca model tepat di lubangnya dan hanya menangkap "
        "cacat mekanis. 'validasi_silang' membangun ulang model tanpa tiap "
        "lubang lalu menebaknya - itu galat ramalan yang sebenarnya.")
    st.dataframe(report.deviation_summary(), width="stretch")
    for line in report.warnings:
        st.warning(line)


def tab_limits(stages) -> None:
    report = stages.limits
    if report is None:
        st.info("Tahap 9 belum dijalankan.")
        return
    if report.blockers:
        st.error(f"{len(report.blockers)} penggugur - status yang DINYATAKAN "
                 "melarang pelaporan di area ini.")
        for line in report.blockers:
            st.write(f"- {line}")
    elif report.readiness:
        st.warning(
            f"{len(report.readiness)} butir kesiapan pelaporan belum dinyatakan. "
            "Estimasi TETAP berjalan - izin dan status lahan tidak mengubah "
            "berapa banyak batubara ada di tanah. Yang tertahan hanya kelengkapan "
            "pernyataan keprospekan beralasan tingkat scoping.")
        for line in report.readiness:
            st.write(f"- {line}")
    else:
        st.success("Legal dan lahan lengkap; pernyataan keprospekan beralasan siap.")
    for line in report.warnings:
        st.warning(line)
    for line in report.notes:
        st.info(line)
    st.dataframe(report.summary(), width="stretch")


def tab_resource(stages) -> None:
    estimate = stages.estimate
    if estimate is None:
        st.info("Tahap 10 belum dijalankan - lihat tab gerbang yang berhenti.")
        return
    label = stages.cfg.resource_label
    st.subheader(f"Tabel {label}")
    frame = estimate.table(label)
    st.dataframe(frame, width="stretch")

    totals = {k: sum(e.tonnes.get(k, 0.0) for e in estimate.estimates)
              for k in ("terukur", "tertunjuk", "tereka", "di luar radius")}
    columns = st.columns(4)
    for column, (key, name) in zip(columns, (
            ("terukur", "Terukur"), ("tertunjuk", "Tertunjuk"),
            ("tereka", "Tereka"), ("di luar radius", "Di luar radius"))):
        column.metric(name, f"{totals[key] / 1e6:,.2f} juta t")
    st.caption(
        "Batubara di luar radius Tereka DIKELUARKAN, tidak pernah dilipat "
        "menjadi Tereka.")
    for line in estimate.warnings:
        st.warning(line)
    with st.expander("Catatan estimasi"):
        for line in estimate.notes:
            st.write(f"- {line}")


def tab_map(stages) -> None:
    if stages.estimate is None or stages.models is None:
        st.info("Peta tersedia setelah tahap 10 selesai.")
        return
    import matplotlib
    matplotlib.use("Agg")
    from coalres.export import bab5

    keys = [e.key for e in stages.estimate.estimates]
    chosen = st.selectbox("Seam", keys)
    estimate = next(e for e in stages.estimate.estimates if e.key == chosen)
    model_obj = stages.models[chosen]

    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    order = ["terukur", "tertunjuk", "tereka", "di luar radius"]
    colors = [bab5.CLASS_COLORS["terukur"], bab5.CLASS_COLORS["tertunjuk"],
              bab5.CLASS_COLORS["tereka"], bab5.OUTSIDE_COLOR]
    coded = np.full(estimate.klass.shape, np.nan)
    for value, name in enumerate(order):
        coded[estimate.klass == name] = value

    figure, ax = plt.subplots(figsize=(8, 6))
    ax.pcolormesh(model_obj.roof.x, model_obj.roof.y, coded,
                  cmap=ListedColormap(colors),
                  norm=BoundaryNorm(np.arange(-0.5, 4.5), 4), shading="auto")
    sub = stages.observation.frame
    sub = sub[(sub["seam"] == estimate.seam) & sub["qualifies"]]
    if len(sub):
        ax.scatter(sub["east"], sub["north"], s=34, c="white",
                   edgecolors="#0b0b0b", linewidths=1.0, zorder=5)
    ax.legend(handles=[Patch(facecolor=c,
                             label=bab5.CLASS_LABELS.get(n, "Di luar radius"))
                       for n, c in zip(order, colors)],
              loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax.set_aspect("equal")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title(f"Peta sumber daya Seam {estimate.seam}")
    st.pyplot(figure)


def tab_export(stages) -> None:
    if not stages.complete:
        st.error(
            "Ekspor tidak tersedia: run berhenti di "
            f"{STAGE_NAMES.get(stages.stopped_at, stages.stopped_at)}. "
            "Gerbang harus diselesaikan lebih dulu - tidak ada jalan pintas.")
        return
    name = st.text_input("Nama proyek", value="SGM")
    folder = st.text_input("Direktori keluaran", value="output")
    if st.button("Tulis kedua paket keluaran", type="primary"):
        with st.spinner("Menulis grid, kontur, tabel, gambar, dan DOCX..."):
            run_obj, problems = pipeline.export_all(stages, folder, name)
        st.success(f"{len(run_obj.files)} berkas ditulis ke `{run_obj.root}`")
        if problems:
            st.error("Swauji penanda gagal:\n\n" + "\n".join(problems))
        else:
            st.info("Swauji penanda _uncut / _ltd lulus.")
        st.dataframe(pd.DataFrame(run_obj.files), width="stretch")


def main() -> None:
    st.set_page_config(page_title="Estimasi Sumber Daya Batubara",
                       layout="wide")
    st.title("Estimasi Sumber Daya Batubara - SNI 5015:2019 / KCMI 2017")

    path = _config_path()
    if not path.exists():
        st.warning(f"Berkas konfigurasi tidak ditemukan: {path}")
        return
    cfg = _load(path)
    if cfg is None:
        return

    spacing = st.sidebar.number_input("Ukuran sel model (m)", 5.0, 100.0, 25.0, 5.0)
    cross = st.sidebar.checkbox("Validasi silang tahap 5", value=True)
    st.session_state["spacing"] = spacing
    st.session_state["cross"] = cross
    st.sidebar.caption(
        "Gerbang tidak dapat dimatikan dari sini. Yang berhenti, berhenti.")

    if st.sidebar.button("Jalankan 11 tahap", type="primary"):
        with st.spinner("Menjalankan..."):
            st.session_state["stages"] = pipeline.run(
                cfg, spacing=spacing, cross_validation=cross)

    stages = st.session_state.get("stages")
    if stages is None:
        st.info("Tekan **Jalankan 11 tahap** di panel kiri.")
        return

    if stages.stopped_at:
        st.error(f"Run BERHENTI di {STAGE_NAMES.get(stages.stopped_at, stages.stopped_at)}")
        for line in stages.messages:
            st.write(f"- {line}")
    else:
        st.success(f"Run selesai. Label keluaran: **{stages.cfg.resource_label}**")

    tabs = st.tabs(["Audit", "Validasi model", "Titik observasi",
                    "Kompleksitas & radius", "Batas KCMI 4.6",
                    "Sumber daya", "Peta", "Ekspor"])
    for tab, render in zip(tabs, (tab_audit, tab_validation, tab_observation,
                                  tab_complexity, tab_limits, tab_resource,
                                  tab_map, tab_export)):
        with tab:
            render(stages)


main()
