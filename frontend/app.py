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


def format_document_label(item: dict) -> str:
    created_raw = item.get("created_at", "")
    try:
        created = datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
        created_text = created.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        created_text = created_raw[:16]

    title = item.get("title", "<No title>")
    return f"#{item['id']} — {title} ({created_text})"


@st.cache_data(ttl=600)
def load_models() -> list[dict]:
    response = requests.get(f"{BACKEND_URL}/models", timeout=20)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=600)
def load_documents() -> list[dict]:
    response = requests.get(f"{BACKEND_URL}/rag/documents", timeout=20)
    response.raise_for_status()
    return response.json()


def load_document_chunks(document_id: int) -> list[dict]:
    response = requests.get(f"{BACKEND_URL}/rag/documents/{document_id}/chunks", timeout=20)
    response.raise_for_status()
    return response.json()


def load_retrieve_results(query: str, limit: int) -> list[dict]:
    response = requests.get(
        f"{BACKEND_URL}/rag/retrieve",
        params={"q": query, "limit": limit},
        timeout=20,
    )
    response.raise_for_status()
    return response.json()


def refresh_missing_embeddings(document_id: int | None = None, limit: int | None = None) -> dict:
    params = {}
    if document_id is not None and document_id > 0:
        params["document_id"] = document_id
    if limit is not None and limit > 0:
        params["limit"] = limit
    response = requests.post(
        f"{BACKEND_URL}/rag/embeddings/refresh",
        params=params,
        timeout=120,
    )
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

tab_ask, tab_history, tab_retrieve, tab_documents = st.tabs(["Hỏi (Ask)", "Lịch sử (History)", "Semantic Retrieval", "Tài liệu (Documents)"])

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

with tab_retrieve:
    st.header("Semantic Retrieval / Semantic Search")
    retrieve_query = st.text_input(
        "Enter query for semantic retrieval",
        value="Why do humans suffer?",
        help="Search the document chunks stored in the RAG database.",
    )
    retrieve_limit = st.slider(
        "Number of results",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
    )

    refresh_col1, refresh_col2 = st.columns([1, 2])
    with refresh_col1:
        refresh_document_id = st.number_input(
            "Document ID to refresh (optional)",
            min_value=0,
            value=0,
            step=1,
            help="Refresh only chunks for a specific document, or leave blank to refresh all missing embeddings.",
        )
    with refresh_col2:
        refresh_limit = st.number_input(
            "Max chunks to refresh",
            min_value=0,
            value=50,
            step=1,
            help="Limit the number of missing chunk embeddings refreshed in one request.",
        )

    if st.button("Refresh missing chunk embeddings", key="refresh_embeddings"):
        try:
            refresh_response = refresh_missing_embeddings(
                document_id=refresh_document_id if refresh_document_id > 0 else None,
                limit=refresh_limit if refresh_limit > 0 else None,
            )
            st.success(
                f"Refreshed {refresh_response.get('refreshed_count', 0)} chunks, "
                f"failed {refresh_response.get('failed_count', 0)}."
            )
            if refresh_response.get("quota_exhausted"):
                st.warning(
                    "Embedding quota exhausted. Please check Gemini billing/quotas and retry later."
                )
            if refresh_response.get("errors"):
                st.error("Some chunks failed to refresh. Check backend logs.")
        except requests.exceptions.RequestException as exc:
            st.error(f"Could not refresh embeddings: {exc}")
        except ValueError:
            st.error("Backend returned invalid JSON during embedding refresh.")

    if st.button("Search chunks", key="retrieve_search"):
        if not retrieve_query.strip():
            st.warning("Please enter a query text for retrieval.")
        else:
            retrieve_response = None
            try:
                results = load_retrieve_results(retrieve_query.strip(), retrieve_limit)
                if not results:
                    st.info("No matching chunks found.")
                else:
                    st.success(f"Found {len(results)} matching chunks.")
                    for chunk in results:
                        score = chunk.get("score")
                        metadata = chunk.get("metadata", {})
                        embedding_model = chunk.get("embedding_model")
                        label = f"Chunk #{chunk.get('id')} - Document #{chunk.get('document_id')}"
                        with st.expander(
                            f"{label} — score: {score:.4f}"
                            if isinstance(score, (int, float))
                            else label
                        ):
                            st.write(chunk.get("content", ""))
                            if embedding_model:
                                st.caption(f"Embedding model: {embedding_model}")
                            st.markdown("**Metadata**")
                            st.json(metadata)
            except requests.exceptions.RequestException as exc:
                st.error(f"Could not retrieve chunks: {exc}")
            except ValueError:
                st.error("Backend returned invalid JSON during retrieval.")

with tab_documents:
    st.header("Quản lý tài liệu RAG / RAG Documents")

    with st.expander("Thêm tài liệu mới / Add new document", expanded=True):
        title = st.text_input("Tiêu đề / Title", value="")
        category = st.text_input("Danh mục / Category", value="Buddhism")
        author = st.text_input("Tác giả / Author", value="")
        source_url = st.text_input("Nguồn / Source URL", value="")
        content = st.text_area(
            "Nội dung / Content",
            value="Đây là nội dung test. Nó sẽ được chia thành nhiều chunk.",
            height=180,
        )

        if st.button("Tạo tài liệu / Create document"):
            if not title.strip() or not category.strip() or not content.strip():
                st.warning("Vui lòng nhập đầy đủ tiêu đề, danh mục và nội dung. / Please enter title, category, and content.")
            else:
                payload = {
                    "title": title.strip(),
                    "category": category.strip(),
                    "author": author.strip() or None,
                    "source_url": source_url.strip() or None,
                    "content": content.strip(),
                }
                response = None
                try:
                    response = requests.post(
                        f"{BACKEND_URL}/rag/documents",
                        json=payload,
                        timeout=30,
                    )
                    response.raise_for_status()
                    document = response.json()
                    st.success(f"Tạo thành công tài liệu #{document.get('id')}.")
                    st.json(document)
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
                    show_backend_error(response, "Không thể tạo tài liệu. / Could not create document.")
                except ValueError:
                    st.error("Backend trả về JSON không hợp lệ. / Backend returned invalid JSON.")

    with st.expander("Tải lên file PDF/TXT / Upload PDF/TXT", expanded=False):
        upload_title = st.text_input("Tiêu đề / Title (file)", value="", key="upload_title")
        upload_category = st.text_input("Danh mục / Category (file)", value="Buddhism", key="upload_category")
        upload_author = st.text_input("Tác giả / Author (file)", value="", key="upload_author")
        upload_source_url = st.text_input("Nguồn / Source URL (file)", value="", key="upload_source_url")
        upload_file = st.file_uploader("Chọn file PDF hoặc TXT / Choose a PDF or TXT file", type=["pdf", "txt"], key="upload_file")

        if st.button("Tải lên và tạo tài liệu / Upload and create document", key="upload_button"):
            if not upload_title.strip() or not upload_category.strip() or upload_file is None:
                st.warning("Vui lòng nhập tiêu đề, danh mục và chọn file. / Please enter title, category, and choose a file.")
            else:
                try:
                    files = {
                        "file": (upload_file.name, upload_file.getvalue(), upload_file.type or "application/octet-stream"),
                    }
                    data = {
                        "title": upload_title.strip(),
                        "category": upload_category.strip(),
                        "author": upload_author.strip() or "",
                        "source_url": upload_source_url.strip() or "",
                    }
                    response = requests.post(
                        f"{BACKEND_URL}/rag/documents/upload",
                        data=data,
                        files=files,
                        timeout=60,
                    )
                    response.raise_for_status()
                    document = response.json()
                    st.success(f"Tải lên và tạo tài liệu thành công #{document.get('id')}.")
                    st.json(document)
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
                    show_backend_error(response, "Không thể tải lên file. / Could not upload the file.")
                except ValueError:
                    st.error("Backend trả về JSON không hợp lệ. / Backend returned invalid JSON.")

    st.write("---")
    st.subheader("Danh sách tài liệu / Document list")

    try:
        documents = load_documents()
        if not documents:
            st.info("Chưa có tài liệu nào. / No documents yet.")
        else:
            labels = {format_document_label(item): item["id"] for item in documents}
            selected_label = st.selectbox(
                "Chọn tài liệu / Select a document",
                list(labels.keys()),
            )
            document_id = labels[selected_label]

            try:
                chunks = load_document_chunks(document_id)
                st.markdown(f"**Tài liệu:** {selected_label}")
                st.write(f"Số chunk: {len(chunks)}")
                for chunk in chunks:
                    with st.expander(f"Chunk #{chunk.get('id')} - Index {chunk.get('metadata', {}).get('chunk_index', '?')}"):
                        st.write(chunk.get("content", ""))
                        embedding_model = chunk.get("embedding_model")
                        if embedding_model:
                            st.caption(f"Embedding model: {embedding_model}")
                        st.markdown("**Metadata**")
                        st.json(chunk.get("metadata", {}))
            except requests.exceptions.RequestException as exc:
                st.error(f"Không tải được chunk của tài liệu: {exc}")
    except requests.exceptions.RequestException:
        st.error(
            "Không kết nối được backend để tải danh sách tài liệu. / "
            "Could not reach the backend to load documents."
        )
