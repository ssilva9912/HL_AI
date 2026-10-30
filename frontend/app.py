import time
from typing import Any

import streamlit as st
from api import (
    Conversation,
    HomelabAPIClient,
    HomelabAPIError,
    IngestionJob,
    QueuedUpload,
    SearchResult,
    Source,
    get_api_url,
    get_default_top_k,
)
from citations import format_citation, group_citations, link_citations
from forms_panel import build_rows, collect_values, summarize
from metrics_panel import (
    DASHBOARDS,
    START_COMMAND,
    dashboard_url,
    get_grafana_health_url,
    get_grafana_url,
    grafana_is_up,
)
from retrieval_notes import describe_correction, looks_like_abstention
from theme import (
    ASSISTANT_AVATAR,
    QUICK_ACTIONS,
    USER_AVATAR,
    header_html,
    inject_styles,
    section_label,
    sidebar_brand_html,
    status_pill,
    welcome_html,
)

API_URL = get_api_url()
DEFAULT_TOP_K = get_default_top_k()


def initialize_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "backend_status" not in st.session_state:
        st.session_state.backend_status = None

    if "ingestion_results" not in st.session_state:
        st.session_state.ingestion_results = []

    if "active_conversation_id" not in st.session_state:
        st.session_state.active_conversation_id = None


def get_api_client() -> HomelabAPIClient:
    return HomelabAPIClient()


def check_backend(
    client: HomelabAPIClient,
) -> None:
    try:
        health = client.health()
    except HomelabAPIError:
        st.session_state.backend_status = None
        return

    st.session_state.backend_status = health


def serialize_source(source: Source) -> dict[str, Any]:
    return {
        "text": source.text,
        "score": source.score,
        "document": source.document,
        "chunk_id": source.chunk_id,
        "citation": source.citation,
        "url": source.url,
    }


def add_message(
    role: str,
    content: str,
    sources: list[dict[str, Any]] | None = None,
) -> None:
    st.session_state.messages.append(
        {
            "role": role,
            "content": content,
            "sources": sources or [],
        }
    )


def load_conversation(
    conversation: Conversation,
) -> None:
    st.session_state.active_conversation_id = conversation.id
    st.session_state.messages = [
        {
            "role": message.role,
            "content": message.content,
            "answer_mode": getattr(
                message,
                "answer_mode",
                "documents",
            ),
            "sources": [serialize_source(source) for source in message.sources],
        }
        for message in conversation.messages
    ]


def serialize_sources(
    result: SearchResult,
) -> list[dict[str, Any]]:
    return [serialize_source(source) for source in result.sources]


def display_sources(
    sources: list[dict[str, Any]],
) -> None:
    citations = group_citations(sources)
    if not citations:
        return

    st.markdown("**Sources**")
    st.markdown("\n".join(f"- {format_citation(citation)}" for citation in citations))

    with st.expander("Show passages", expanded=False):
        for citation in citations:
            st.markdown(f"**[{citation.number}] {citation.label}**")
            for passage in citation.passages:
                metadata_parts = [f"Score: `{float(passage.get('score', 0.0)):.4f}`"]
                if passage.get("chunk_id"):
                    metadata_parts.append(f"Chunk: `{passage['chunk_id']}`")
                st.caption(" · ".join(metadata_parts))
                st.write(passage.get("text", ""))
            if citation is not citations[-1]:
                st.divider()


def display_answer_mode(answer_mode: str, answer: str = "") -> None:
    if answer_mode in {"documents", "web", "documents_web"} and looks_like_abstention(answer):
        st.caption("The sources found didn't answer this · try rephrasing or check the links below")
    elif answer_mode == "general":
        st.caption(
            "General local-model answer · not grounded in uploaded documents",
        )
    elif answer_mode == "web":
        st.caption("Answered from web pages · check the cited sources")
    elif answer_mode == "documents_web":
        st.caption("Answered from your documents and web pages")
    else:
        st.caption("Document-grounded answer")


def display_chat_history() -> None:
    for message in st.session_state.messages:
        role = message.get(
            "role",
            "assistant",
        )
        content = message.get(
            "content",
            "",
        )
        sources = message.get(
            "sources",
            [],
        )
        answer_mode = message.get(
            "answer_mode",
            "documents",
        )

        with st.chat_message(role, avatar=USER_AVATAR if role == "user" else ASSISTANT_AVATAR):
            st.markdown(link_citations(content, group_citations(sources)))

            if role == "assistant":
                display_answer_mode(answer_mode, content)
                display_sources(sources)


def render_sidebar(
    client: HomelabAPIClient,
) -> int:
    with st.sidebar:
        st.markdown(sidebar_brand_html(), unsafe_allow_html=True)

        if st.button(
            "New conversation",
            icon=":material/add:",
            use_container_width=True,
            type="primary",
        ):
            try:
                conversation = client.create_conversation()
            except HomelabAPIError as exc:
                st.error(str(exc))
            else:
                load_conversation(conversation)
                st.rerun()

        try:
            conversations = client.list_conversations()
        except HomelabAPIError as exc:
            conversations = []
            st.warning(f"Could not load conversations: {exc}")

        if conversations:
            st.markdown(section_label("Conversations"), unsafe_allow_html=True)
            conversation_by_id = {conversation.id: conversation for conversation in conversations}
            active_id = st.session_state.active_conversation_id
            if active_id not in conversation_by_id:
                active_id = conversations[0].id
                load_conversation(
                    client.get_conversation(active_id),
                )

            conversation_ids = list(conversation_by_id)
            selected_id = st.selectbox(
                "Saved conversations",
                options=conversation_ids,
                index=conversation_ids.index(active_id),
                format_func=lambda conversation_id: conversation_by_id[conversation_id].title,
                label_visibility="collapsed",
            )
            if selected_id != st.session_state.active_conversation_id:
                try:
                    load_conversation(
                        client.get_conversation(selected_id),
                    )
                except HomelabAPIError as exc:
                    st.error(str(exc))
                else:
                    st.rerun()

            active = conversation_by_id[selected_id]
            with st.expander("Rename or delete"):
                title = st.text_input(
                    "Title",
                    value=active.title,
                    key=f"title-{active.id}",
                )
                if st.button(
                    "Rename",
                    key=f"rename-{active.id}",
                    use_container_width=True,
                ):
                    try:
                        renamed = client.rename_conversation(
                            active.id,
                            title,
                        )
                    except (ValueError, HomelabAPIError) as exc:
                        st.error(str(exc))
                    else:
                        load_conversation(renamed)
                        st.rerun()
                if st.button(
                    "Delete conversation",
                    key=f"delete-{active.id}",
                    use_container_width=True,
                ):
                    try:
                        client.delete_conversation(active.id)
                    except HomelabAPIError as exc:
                        st.error(str(exc))
                    else:
                        st.session_state.active_conversation_id = None
                        st.session_state.messages = []
                        st.rerun()
        else:
            st.caption("Start a conversation to begin chatting.")

        st.markdown(section_label("Answer settings"), unsafe_allow_html=True)

        top_k = st.slider(
            label="Sources per answer",
            min_value=1,
            max_value=20,
            value=DEFAULT_TOP_K,
            step=1,
            help=("How many of the best-matching passages are used for each answer."),
        )

        st.toggle(
            "Search the web when needed",
            key="web_search",
            help=(
                "Allows an online search when your documents don't cover a question. "
                "Only then is the question sent to the configured search engine, and "
                "page text is treated as untrusted."
            ),
        )

        st.markdown(section_label("Connection"), unsafe_allow_html=True)

        status = st.session_state.backend_status
        st.markdown(status_pill(status is not None), unsafe_allow_html=True)

        with st.expander("Details"):
            if status is None:
                st.caption(f"Can't reach the backend. Start it on `{API_URL}`.")
            else:
                st.caption(f"Service: `{status.get('service', 'homelab-ai')}`")
            st.caption(f"API: `{API_URL}`")
            if st.button(
                "Check again",
                icon=":material/refresh:",
                use_container_width=True,
            ):
                check_backend(client)
                st.rerun()

    return int(top_k)


def render_chat_tab(
    client: HomelabAPIClient,
    top_k: int,
    pages: dict[str, Any],
) -> None:
    welcome = st.empty()
    if not st.session_state.messages:
        with welcome.container():
            st.markdown(welcome_html(), unsafe_allow_html=True)
            columns = st.columns(len(QUICK_ACTIONS))
            for column, (label, icon, target) in zip(columns, QUICK_ACTIONS, strict=True):
                if column.button(
                    label,
                    key=f"quick_action_{target}",
                    icon=icon,
                    use_container_width=True,
                ):
                    st.switch_page(pages[target])

    display_chat_history()

    question = st.chat_input("Ask anything about your documents...")

    if question is None:
        return

    cleaned_question = question.strip()

    if not cleaned_question:
        st.warning("Enter a question before submitting.")
        return

    welcome.empty()

    conversation_id = st.session_state.active_conversation_id
    if conversation_id is None:
        try:
            conversation = client.create_conversation()
        except HomelabAPIError as exc:
            st.error(f"Could not create a conversation: {exc}")
            return
        load_conversation(conversation)
        conversation_id = conversation.id

    add_message(
        role="user",
        content=cleaned_question,
    )

    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(cleaned_question)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        web_enabled = bool(st.session_state.get("web_search", False))
        note_placeholder = st.empty()
        answer_placeholder = st.empty()
        answer = ""
        citations: list[Any] = []
        stream_completed = False
        try:
            for event in client.stream_conversation_message(
                conversation_id=conversation_id,
                content=cleaned_question,
                top_k=top_k,
                web=web_enabled,
            ):
                event_type = str(event.get("type", ""))
                if event_type == "start":
                    citations = group_citations(list(event.get("sources") or []))
                    note = describe_correction(event.get("correction"), web_enabled)
                    if note:
                        note_placeholder.caption(note)
                elif event_type == "token":
                    answer += str(event.get("content", ""))
                    answer_placeholder.markdown(f"{link_citations(answer, citations)} _")
                elif event_type == "complete":
                    stream_completed = True
        except (ValueError, HomelabAPIError) as exc:
            answer_placeholder.empty()
            st.error(f"Streaming failed: {exc}")
            st.session_state.backend_status = None
            try:
                load_conversation(client.get_conversation(conversation_id))
            except HomelabAPIError:
                pass
            return

        if not stream_completed:
            answer_placeholder.empty()
            st.error("The response stream ended before completion. You can retry.")
            return

        answer_placeholder.markdown(link_citations(answer, citations))
        try:
            conversation = client.get_conversation(conversation_id)
        except HomelabAPIError as exc:
            st.error(f"The answer completed but could not be reloaded: {exc}")
            return

        load_conversation(conversation)
        assistant_messages = [
            message for message in conversation.messages if message.role == "assistant"
        ]
        if assistant_messages:
            assistant = assistant_messages[-1]
            display_answer_mode(assistant.answer_mode, assistant.content)
            display_sources([serialize_source(source) for source in assistant.sources])


def render_documents_tab(
    client: HomelabAPIClient,
) -> None:
    st.subheader("Documents")
    st.caption(
        "Queue documents for durable background parsing, embedding, and indexing.",
    )

    with st.container(border=True):
        st.markdown("#### Read-only network share")
        st.caption(
            "Sync now inventories supported files. It does not ingest, modify, or "
            "delete files yet.",
        )
        if st.button("Sync now", key="sync_network_share"):
            try:
                scan = client.sync_network_share()
            except HomelabAPIError as exc:
                st.error(f"Network-share scan failed: {exc}")
            else:
                st.success(f"Scanned {scan.source_name} ({scan.root_path}).")
                st.write(
                    f"New: **{scan.discovered}** · Changed: **{scan.changed}** · "
                    f"Unchanged: **{scan.unchanged}** · Missing: **{scan.missing}**"
                )
                if scan.unsupported or scan.unsafe:
                    st.caption(
                        f"Skipped unsupported: {scan.unsupported} · "
                        f"Skipped unsafe links: {scan.unsafe}"
                    )

    uploaded_files = st.file_uploader(
        label="Choose documents",
        type=[
            "txt",
            "md",
            "pdf",
        ],
        accept_multiple_files=True,
        help=("Supported formats: text, Markdown, and text-based PDF files."),
    )

    if uploaded_files:
        st.write(f"Selected files: **{len(uploaded_files)}**")

        for uploaded_file in uploaded_files:
            st.caption(f"{uploaded_file.name} ({uploaded_file.size:,} bytes)")

    upload_clicked = st.button(
        "Queue uploads",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_files,
    )

    if upload_clicked:
        results: list[tuple[str, IngestionJob]] = []
        submission_failures = 0
        total_files = len(uploaded_files)

        with st.status(
            "Processing queued documents...",
            expanded=True,
        ) as batch_status:
            for index, uploaded_file in enumerate(
                uploaded_files,
                start=1,
            ):
                st.write(
                    f"**{index}/{total_files}:** Queuing `{uploaded_file.name}`...",
                )

                try:
                    queued = client.queue_document(
                        filename=uploaded_file.name,
                        content=uploaded_file.getvalue(),
                        content_type=uploaded_file.type,
                    )
                    job = poll_ingestion_job(
                        client=client,
                        queued=queued,
                        document_name=uploaded_file.name,
                    )
                except (
                    ValueError,
                    HomelabAPIError,
                ) as exc:
                    submission_failures += 1
                    st.error(f"{uploaded_file.name}: {exc}")
                else:
                    results.append((uploaded_file.name, job))

            failed_count = sum(job.status == "failed" for _, job in results)
            total_failures = failed_count + submission_failures
            if total_failures:
                batch_status.update(
                    label=(f"Document processing finished with {total_failures} error(s)."),
                    state="error",
                    expanded=True,
                )
            else:
                batch_status.update(
                    label=(f"Successfully indexed {len(results)} document(s)."),
                    state="complete",
                    expanded=False,
                )

        st.session_state.ingestion_results = results

    if not upload_clicked and not st.session_state.ingestion_results:
        st.info("Select one or more documents, then click Queue uploads.")

    render_ingestion_results(client)
    render_document_library(client)


def poll_ingestion_job(
    *,
    client: HomelabAPIClient,
    queued: QueuedUpload,
    document_name: str,
    timeout_seconds: float = 900.0,
) -> IngestionJob:
    stage_progress = {
        "queued": 0.05,
        "parsing": 0.25,
        "embedding": 0.60,
        "indexing": 0.85,
        "succeeded": 1.0,
        "failed": 1.0,
    }
    progress = st.progress(0.0, text=f"`{document_name}` queued")
    deadline = time.monotonic() + timeout_seconds

    while True:
        job = client.get_ingestion_job(queued.job_id)
        total = job.total_chunks
        chunk_text = f" · {job.processed_chunks}/{total} chunks" if total is not None else ""
        progress.progress(
            stage_progress.get(job.stage, 0.05),
            text=f"`{document_name}` — {job.stage}{chunk_text}",
        )

        if job.status == "succeeded":
            st.success(
                f"Indexed `{document_name}` with {job.processed_chunks} chunk(s).",
            )
            return job

        if job.status == "failed":
            st.error(
                f"`{document_name}` failed: {job.error_message or 'Unknown ingestion error.'}",
            )
            return job

        if time.monotonic() >= deadline:
            raise HomelabAPIError(
                f"Timed out waiting for `{document_name}`. The durable job may still be running.",
            )

        time.sleep(0.5)


def render_ingestion_results(
    client: HomelabAPIClient,
) -> None:
    results: list[tuple[str, IngestionJob]] = st.session_state.ingestion_results
    if not results:
        return

    st.divider()
    st.markdown("### Latest ingestion jobs")
    st.dataframe(
        [
            {
                "Document": name,
                "Status": job.status,
                "Stage": job.stage,
                "Attempt": job.attempt_count,
                "Chunks": (
                    f"{job.processed_chunks}/{job.total_chunks}"
                    if job.total_chunks is not None
                    else str(job.processed_chunks)
                ),
                "Error": job.error_message or "",
            }
            for name, job in results
        ],
        use_container_width=True,
        hide_index=True,
    )

    for name, job in results:
        if job.status != "failed":
            continue
        if st.button(
            f"Retry {name}",
            key=f"retry-{job.id}",
            use_container_width=True,
        ):
            try:
                queued = client.retry_ingestion_job(job.id)
                retried = poll_ingestion_job(
                    client=client,
                    queued=queued,
                    document_name=name,
                )
            except HomelabAPIError as exc:
                st.error(f"Retry failed: {exc}")
            else:
                st.session_state.ingestion_results = [
                    (existing_name, retried if existing_job.id == job.id else existing_job)
                    for existing_name, existing_job in results
                ]
                st.rerun()


def render_document_library(
    client: HomelabAPIClient,
) -> None:
    st.divider()
    st.markdown("### Document library")
    try:
        documents = client.list_documents()
    except HomelabAPIError as exc:
        st.warning(str(exc))
        return

    if not documents:
        st.info("No documents have been indexed yet.")
        return

    st.dataframe(
        [
            {
                "Document": document.filename,
                "Type": document.content_type or "",
                "Size (bytes)": document.size_bytes,
                "Status": document.status,
                "Chunks": document.chunk_count,
                "Error": document.error_message or "",
            }
            for document in documents
        ],
        use_container_width=True,
        hide_index=True,
    )


def render_forms_tab(client: HomelabAPIClient) -> None:
    st.subheader("Forms")
    st.caption(
        "Upload a Word (.docx) or fillable PDF form. Suggested answers come only from your own "
        "documents, never from the web. Review everything before you use the filled form."
    )

    uploaded = st.file_uploader("Form to fill", type=["docx", "pdf"], key="form_upload")
    if uploaded is None:
        return

    content = uploaded.getvalue()
    form_key = f"{uploaded.name}:{len(content)}"

    if st.button("Find fields and suggest answers", type="primary"):
        with st.spinner("Reading the form and searching your documents..."):
            try:
                st.session_state.form_proposal = (
                    form_key,
                    client.propose_form(uploaded.name, content),
                )
            except HomelabAPIError as exc:
                st.error(str(exc))
                return
        st.session_state.pop("filled_form", None)

    stored = st.session_state.get("form_proposal")
    if not stored or stored[0] != form_key:
        return
    proposal = stored[1]

    st.info(summarize(proposal.fields))
    rows = st.data_editor(
        build_rows(proposal.fields),
        hide_index=True,
        use_container_width=True,
        disabled=["Field", "Check"],
        key=f"form_editor_{form_key}",
    )

    ticked = {
        field.id: st.checkbox(field.label, key=f"form_check_{form_key}_{field.id}")
        for field in proposal.fields
        if field.kind == "checkbox"
    }

    if st.button("Create filled form"):
        values = collect_values(proposal.fields, list(rows), ticked)
        try:
            st.session_state.filled_form = (
                form_key,
                client.fill_form(uploaded.name, content, values),
            )
        except HomelabAPIError as exc:
            st.error(str(exc))
            return

    filled = st.session_state.get("filled_form")
    if filled and filled[0] == form_key:
        stem, _, extension = uploaded.name.rpartition(".")
        st.download_button(
            "Download filled form",
            data=filled[1],
            file_name=f"{stem or uploaded.name}-filled.{extension.lower()}",
            type="primary",
        )


def render_metrics_tab() -> None:
    st.subheader("Metrics")
    st.caption("Live speed and quality of the language model and the retrieval pipeline.")

    if not grafana_is_up(get_grafana_health_url()):
        st.info(f"The metrics dashboards are not running. Start them with:\n\n`{START_COMMAND}`")
        return

    base_url = get_grafana_url()
    selected = st.segmented_control(
        "Dashboard",
        options=list(DASHBOARDS),
        default=next(iter(DASHBOARDS)),
        key="metrics_dashboard",
        label_visibility="collapsed",
    )
    uid = DASHBOARDS[selected or next(iter(DASHBOARDS))]

    st.link_button(
        "Open in Grafana",
        dashboard_url(base_url, uid),
        help="Opens the full dashboard in a new tab.",
    )
    st.iframe(dashboard_url(base_url, uid, embedded=True), height=900)


def main() -> None:
    st.set_page_config(
        page_title="Homelab AI",
        page_icon="🏠",
        layout="wide",
    )

    initialize_session_state()
    inject_styles()

    client = get_api_client()

    if st.session_state.backend_status is None:
        check_backend(client)

    top_k = render_sidebar(client)

    st.markdown(header_html(st.session_state.backend_status is not None), unsafe_allow_html=True)

    # Pages rather than tabs: a chat input inside a tab floats mid-page, while one in the main
    # area stays docked at the bottom of the screen.
    pages: dict[str, Any] = {}
    pages["chat"] = st.Page(
        lambda: render_chat_tab(client=client, top_k=top_k, pages=pages),
        title="Chat",
        icon=":material/chat:",
        default=True,
    )
    pages["documents"] = st.Page(
        lambda: render_documents_tab(client=client),
        title="Documents",
        icon=":material/folder_open:",
        url_path="documents",
    )
    pages["forms"] = st.Page(
        lambda: render_forms_tab(client=client),
        title="Forms",
        icon=":material/edit_document:",
        url_path="forms",
    )
    pages["metrics"] = st.Page(
        render_metrics_tab,
        title="Metrics",
        icon=":material/monitoring:",
        url_path="metrics",
    )

    st.navigation(list(pages.values()), position="top").run()


if __name__ == "__main__":
    main()
