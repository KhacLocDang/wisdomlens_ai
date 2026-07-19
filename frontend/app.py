import os
from datetime import datetime

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

LANGUAGE_OPTIONS = {
    "Tiếng Việt": "vi",
    "English": "en",
}


def render_answer(data: dict) -> None:
    st.subheader("Tóm tắt / Summary")
    st.write(data.get("summary", ""))

    st.subheader("Phật giáo / Buddhism")
    st.write(data.get("buddhism", ""))

    st.subheader("Triết học phương Tây / Western Philosophy")
    st.write(data.get("western_philosophy", ""))

    st.subheader("Tâm lý học / Psychology")
    st.write(data.get("psychology", ""))

    st.subheader("Điểm tương đồng / Similarities")
    st.write(data.get("similarities", ""))

    st.subheader("Điểm khác biệt / Differences")
    st.write(data.get("differences", ""))

    st.subheader("Tài liệu tham khảo / References")
    refs = data.get("references") or []
    if refs:
        for ref in refs:
            st.markdown(f"- {ref}")
    else:
        st.write("Chưa có tài liệu tham khảo. / No references yet.")


def show_backend_error(response: requests.Response | None, fallback: str) -> None:
    if response is not None:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text or fallback
        st.error(f"Lỗi backend / Backend error ({response.status_code}): {detail}")
    else:
        st.error(fallback)


def format_inquiry_label(item: dict) -> str:
    created_raw = item.get("created_at", "")
    try:
        created = datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
        created_text = created.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        created_text = created_raw[:16]

    question = item.get("question", "")
    if len(question) > 80:
        question = question[:77] + "..."

    lang = item.get("language", "vi").upper()
    return f"#{item['id']} [{lang}] — {question} ({created_text})"


@st.cache_data(ttl=600)
def load_models() -> list[dict]:
    response = requests.get(f"{BACKEND_URL}/models", timeout=20)
    response.raise_for_status()
    return response.json()


st.set_page_config(page_title="WisdomLens AI", page_icon="🧠", layout="wide")

st.title("WisdomLens AI")
st.markdown(
    "Khám phá câu hỏi cuộc sống qua **Phật giáo**, **Triết học phương Tây** "
    "và **Tâm lý học**. / Explore life questions through **Buddhism**, "
    "**Western philosophy**, and **psychology**. "
    "Đây là góc nhìn có cấu trúc — không phải tư vấn cá nhân hay trị liệu. / "
    "Structured perspectives — not personal advice or therapy."
)

tab_ask, tab_history = st.tabs(["Hỏi (Ask)", "Lịch sử (History)"])

with tab_ask:
    col_lang, col_model = st.columns(2)

    with col_lang:
        selected_lang_label = st.selectbox(
            "Ngôn ngữ trả lời / Answer language",
            list(LANGUAGE_OPTIONS.keys()),
        )
        language = LANGUAGE_OPTIONS[selected_lang_label]

    model_id = None
    with col_model:
        try:
            models = load_models()
            if models:
                model_labels = {
                    f"{m.get('display_name') or m['id']} ({m['id']})": m["id"]
                    for m in models
                }
                # Prefer gemini-2.5-flash as default when present
                default_index = 0
                for i, mid in enumerate(model_labels.values()):
                    if mid == "gemini-2.5-flash":
                        default_index = i
                        break
                selected_model_label = st.selectbox(
                    "Mô hình / Model",
                    list(model_labels.keys()),
                    index=default_index,
                )
                model_id = model_labels[selected_model_label]
            else:
                st.warning("Không có model nào. / No models available.")
        except requests.exceptions.RequestException:
            st.warning(
                "Không tải được danh sách model — sẽ dùng mặc định từ backend. / "
                "Could not load models — backend default will be used."
            )

    question = st.text_area(
        "Câu hỏi của bạn / Your question",
        placeholder="Vì sao con người sợ thất bại? / Why are humans afraid of failure?",
        height=120,
    )

    if st.button("Hỏi WisdomLens / Ask WisdomLens", type="primary"):
        if not question.strip():
            st.warning("Vui lòng nhập câu hỏi. / Please enter a question first.")
        else:
            response = None
            payload = {"question": question.strip(), "language": language}
            if model_id:
                payload["model"] = model_id
            try:
                response = requests.post(
                    f"{BACKEND_URL}/ask",
                    json=payload,
                    timeout=90,
                )
                response.raise_for_status()
                render_answer(response.json())
            except requests.exceptions.ConnectionError:
                st.error(
                    "Không kết nối được backend. Hãy chạy Docker Compose. / "
                    "Could not reach the backend. "
                    f"Make sure FastAPI is running at `{BACKEND_URL}`."
                )
            except requests.exceptions.Timeout:
                st.error(
                    "Backend phản hồi quá lâu. Vui lòng thử lại. / "
                    "The backend took too long to respond. Please try again."
                )
            except requests.exceptions.HTTPError:
                show_backend_error(response, "Không thể lấy câu trả lời. / Could not get an answer.")
            except ValueError:
                st.error("Backend trả về JSON không hợp lệ. / Backend returned invalid JSON.")

with tab_history:
    if "history_query" not in st.session_state:
        st.session_state.history_query = ""

    search_text = st.text_input(
        "Tìm câu hỏi đã lưu / Search saved questions",
        value=st.session_state.history_query,
    )
    if st.button("Tìm / Search"):
        st.session_state.history_query = search_text.strip()

    params = {"limit": 20}
    if st.session_state.history_query:
        params["q"] = st.session_state.history_query

    response = None
    try:
        response = requests.get(f"{BACKEND_URL}/inquiries", params=params, timeout=10)
        response.raise_for_status()
        inquiries = response.json()

        if not inquiries:
            st.info(
                "Không tìm thấy. / No results."
            )
        else:
            labels = {format_inquiry_label(item): item["id"] for item in inquiries}
            selected_label = st.selectbox(
                "Chọn câu hỏi đã lưu / Select a saved question",
                list(labels.keys()),
            )
            inquiry_id = labels[selected_label]

            detail_response = None
            try:
                detail_response = requests.get(
                    f"{BACKEND_URL}/inquiries/{inquiry_id}",
                    timeout=10,
                )
                detail_response.raise_for_status()
            except requests.exceptions.HTTPError:
                show_backend_error(detail_response, "Không tải được chi tiết. / Could not load detail.")
                st.stop()

            detail = detail_response.json()

            st.markdown(f"**Câu hỏi / Question:** {detail.get('question', '')}")
            meta_parts = [f"Nguồn / Source: {detail.get('source', '')}"]
            lang_code = detail.get("language", "")
            if lang_code:
                lang_display = "Tiếng Việt" if lang_code == "vi" else "English"
                meta_parts.append(f"Ngôn ngữ / Language: {lang_display}")
            if detail.get("model"):
                meta_parts.append(f"Model: {detail['model']}")
            if detail.get("created_at"):
                meta_parts.append(f"Thời gian / Time: {detail['created_at']}")
            st.caption(" | ".join(meta_parts))

            render_answer(detail)

    except requests.exceptions.ConnectionError:
        st.error(
            "Không kết nối được backend. Hãy chạy Docker Compose. / "
            "Could not reach the backend. "
            f"Make sure FastAPI is running at `{BACKEND_URL}`."
        )
    except requests.exceptions.Timeout:
        st.error(
            "Backend phản hồi quá lâu. Vui lòng thử lại. / "
            "The backend took too long to respond. Please try again."
        )
    except requests.exceptions.HTTPError:
        show_backend_error(response, "Không thể tải lịch sử. / Could not load history.")
