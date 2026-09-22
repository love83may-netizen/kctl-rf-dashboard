import base64
import io
import json
import math
import os
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from docxtpl import DocxTemplate
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


@st.cache_data(show_spinner="🔍 AI Vision이 Mkr1 마커를 인식하는 중...")
def extract_marker_data(image_bytes: bytes, api_key: str) -> dict:
    """스펙트럼 분석기 화면 우측 상단 Mkr1(녹색 텍스트) 영역을 인식하여
    주파수/전력 마커 값을 JSON으로 반환한다."""
    client = OpenAI(api_key=api_key)
    b64_str = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:image/png;base64,{b64_str}"

    response = client.chat.completions.create(
        model="gpt-5.6-luna",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "너는 Agilent Swept SA 스펙트럼 분석기 화면을 판독하는 RF 계측 전문가다. "
                    "화면 우측 상단의 녹색 텍스트로 표시된 Mkr1 마커의 주파수와 전력 값을 읽어 "
                    '반드시 다음 JSON 스키마로만 응답하라: '
                    '{"freq_val": <number>, "freq_unit": "GHz 또는 MHz", '
                    '"power_val": <number>, "power_unit": "uW"}'
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "이 이미지의 Mkr1 마커 값을 JSON으로 추출해줘."},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
    )
    return json.loads(response.choices[0].message.content)


def calc_freq_mhz(freq_val: float, freq_unit: str) -> float:
    if freq_unit == "GHz":
        return round(freq_val * 1000, 2)
    return round(freq_val, 2)


def calc_power_dbm(power_uw: float) -> float:
    return round(10 * math.log10(power_uw / 1000), 2)


def calc_field_strength_dbuv(power_dbm: float, cf_db: float) -> float:
    return round(power_dbm + 107.0 + cf_db, 2)


def calc_margin(limit_dbuv: float, field_strength_dbuv: float) -> float:
    return round(limit_dbuv - field_strength_dbuv, 2)


def judge_verdict(margin: float) -> str:
    return "PASS" if margin >= 0 else "FAIL"


def build_analysis_result(no: int, image_name: str, marker: dict, cf_db: float, limit_dbuv: float) -> dict:
    freq_mhz = calc_freq_mhz(marker["freq_val"], marker["freq_unit"])
    power_uw = round(marker["power_val"], 2)
    power_dbm = calc_power_dbm(power_uw)
    dbuv_m = calc_field_strength_dbuv(power_dbm, cf_db)
    margin = calc_margin(limit_dbuv, dbuv_m)
    return {
        "no": no,
        "image_name": image_name,
        "freq_mhz": freq_mhz,
        "power_uw": power_uw,
        "dbm": power_dbm,
        "dbuv_m": dbuv_m,
        "limit": limit_dbuv,
        "margin": margin,
        "verdict": judge_verdict(margin),
    }

st.set_page_config(page_title="Eurofins KCTL RF Inspector", page_icon="📡", layout="wide")

with st.sidebar:
    st.title("📡 KCTL RF 계측 Inspector")

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
    )
    st.caption("AI Engine: OpenAI 5.6 Luna Vision")

    tester = st.text_input("시험 담당자", value="홍길동 선임연구원")
    reviewer = st.text_input("기술 검토자", value="김선임 기술책임자")
    sample_name = st.text_input("시료명(EUT)", value="EUT-2026-BLE-MODULE")

    standard = st.selectbox(
        "시험 규격 선택",
        ["FCC Part 15 Subpart B Class B (3 m)", "CISPR 32 Class B (3 m)", "KN 32"],
    )

    cf_db = st.slider("안테나 보정계수 (CF)", min_value=15.0, max_value=40.0, value=28.5, step=0.5)
    limit_dbuv = st.slider("규격 기준치 (Limit)", min_value=120.0, max_value=150.0, value=140.0, step=1.0)

st.title("📡 Eurofins KCTL RF 계측 Inspector")
st.markdown("스펙트럼 분석기 이미지를 업로드하면 AI가 마커 데이터를 인식하여 방사성 방출(RE) 시험 결과를 자동 산출합니다.")

if "target_images" not in st.session_state:
    st.session_state["target_images"] = []

uploaded_files = st.file_uploader(
    "스펙트럼 분석기 이미지 업로드",
    accept_multiple_files=True,
    type=["png", "jpg"],
)

if uploaded_files:
    st.session_state["target_images"] = [
        {"name": f.name, "bytes": f.getvalue()} for f in uploaded_files
    ]

if st.button("📂 기본 샘플 3종(img_1~3) 일괄 불러오기"):
    sample_names = ["img_1.png", "img_2.png", "img_3.png"]
    loaded = []
    for name in sample_names:
        path = os.path.join(os.path.dirname(__file__), name)
        with open(path, "rb") as f:
            loaded.append({"name": name, "bytes": f.read()})
    st.session_state["target_images"] = loaded
    st.success(f"샘플 이미지 {len(loaded)}개를 불러왔습니다.")

if st.session_state["target_images"]:
    st.caption(f"분석 대기 이미지: {len(st.session_state['target_images'])}개")
    st.image(
        [img["bytes"] for img in st.session_state["target_images"]],
        caption=[img["name"] for img in st.session_state["target_images"]],
        width=220,
    )

    if st.button("🔍 AI 마커 인식 시작", type="primary"):
        if not api_key:
            st.warning("사이드바에 OpenAI API Key를 입력하세요.")
            st.stop()

        try:
            markers = []
            for img in st.session_state["target_images"]:
                marker = extract_marker_data(img["bytes"], api_key)
                markers.append({"name": img["name"], "marker": marker})
            st.session_state["marker_data"] = markers
            st.success(f"총 {len(markers)}건의 마커 인식이 완료되었습니다.")
        except Exception as e:
            st.error(f"오류 내용: {e}")

# CF/Limit 슬라이더가 바뀌어도 캐시된 마커 데이터로 즉시 재계산할 뿐 API를 재호출하지 않는다.
if st.session_state.get("marker_data"):
    st.session_state["analysis_results"] = [
        build_analysis_result(idx, m["name"], m["marker"], cf_db, limit_dbuv)
        for idx, m in enumerate(st.session_state["marker_data"], start=1)
    ]

if st.session_state.get("analysis_results"):
    results = st.session_state["analysis_results"]
    df = pd.DataFrame(results)

    total = len(results)
    pass_cnt = int((df["verdict"] == "PASS").sum())
    fail_cnt = int((df["verdict"] == "FAIL").sum())
    overall_verdict = "FAIL" if fail_cnt > 0 else "PASS"

    st.subheader("📊 측정 결과 요약")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("총 측정 건수", total)
    col2.metric("PASS 건수", pass_cnt, delta=f"{pass_cnt}건", delta_color="normal")
    col3.metric("FAIL 건수", fail_cnt, delta=f"{fail_cnt}건", delta_color="inverse")
    col4.metric(
        "종합 판정",
        overall_verdict,
        delta="적합" if overall_verdict == "PASS" else "부적합",
        delta_color="normal" if overall_verdict == "PASS" else "inverse",
    )

    st.subheader("📈 규격 비교 차트")
    bar_colors = ["#003399" if v == "PASS" else "#E03A3A" for v in df["verdict"]]
    fig = go.Figure(
        go.Bar(
            x=df["freq_mhz"],
            y=df["dbuv_m"],
            marker_color=bar_colors,
            customdata=df[["image_name", "margin"]],
            hovertemplate=(
                "이미지: %{customdata[0]}<br>"
                "주파수: %{x} MHz<br>"
                "측정값: %{y} dBµV/m<br>"
                "마진: %{customdata[1]} dB<extra></extra>"
            ),
            name="측정값",
        )
    )
    fig.add_hline(
        y=limit_dbuv,
        line_dash="dash",
        line_color="red",
        annotation_text="FCC Limit 기준",
        annotation_position="top left",
    )
    fig.update_layout(
        xaxis_title="주파수 (MHz)",
        yaxis_title="측정 전계강도 (dBµV/m)",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 판정 결과 상세")
    display_df = df.rename(
        columns={
            "no": "No",
            "image_name": "이미지명",
            "freq_mhz": "주파수(MHz)",
            "power_uw": "측정전력(µW)",
            "dbm": "dBm",
            "dbuv_m": "측정값(dBµV/m)",
            "limit": "Limit",
            "margin": "마진(dB)",
            "verdict": "판정",
        }
    )[["No", "이미지명", "주파수(MHz)", "측정전력(µW)", "dBm", "측정값(dBµV/m)", "Limit", "마진(dB)", "판정"]]
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.subheader("📥 공식 시험성적서 다운로드")
    remarks = st.text_area("특이사항", value="이상 없음", height=80)

    try:
        template_path = os.path.join(os.path.dirname(__file__), "report_template.docx")
        tpl = DocxTemplate(template_path)

        now = datetime.now()
        context = {
            "doc_no": f"KCTL-RE-{now:%Y%m%d}-{now:%H%M%S}",
            "test_date": now.strftime("%Y-%m-%d"),
            "test_name": "방사성 방출 (RE) 측정 결과 보고서",
            "tester": tester,
            "standard": standard,
            "sample_name": sample_name,
            "cf_db": cf_db,
            "reviewer": reviewer,
            "remarks": remarks,
            "generated_at": now.strftime("%Y-%m-%d %H:%M"),
            "total": total,
            "pass_cnt": pass_cnt,
            "fail_cnt": fail_cnt,
            "verdict": overall_verdict,
            "rows": [
                {
                    "no": r["no"],
                    "freq_mhz": r["freq_mhz"],
                    "power_uw": r["power_uw"],
                    "dbm": r["dbm"],
                    "dbuv_m": r["dbuv_m"],
                    "limit": r["limit"],
                    "margin": r["margin"],
                    "verdict": r["verdict"],
                }
                for r in results
            ],
        }
        tpl.render(context)

        buffer = io.BytesIO()
        tpl.save(buffer)
        buffer.seek(0)

        st.download_button(
            label="📥 공식 시험성적서(.docx) 다운로드",
            data=buffer,
            file_name=f"KCTL_RE_Report_{now:%Y%m%d}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    except Exception as e:
        st.error(f"오류 내용: {e}")
