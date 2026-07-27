import time
from typing import Any

import streamlit as st
from api import (
    Conversation,
    EvaluationResult,
    HomelabAPIClient,
    HomelabAPIError,
    IngestionJob,
    QueuedUpload,
    SearchResult,
    get_api_url,
    get_default_top_k,
)

API_URL = get_api_url()
DEFAULT_TOP_K = get_default_top_k()


def initialize_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "backend_status" not in st.session_state:
        st.session_state.backend_status = None

    if "evaluation_result" not in st.session_state:
        st.session_state.evaluation_result = None

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
            "sources": [
                {
                    "text": source.text,
                    "score": source.score,
                    "document": source.document,
                    "chunk_id": source.chunk_id,
                }
                for source in message.sources
            ],
        }
        for message in conversation.messages
    ]


def serialize_sources(
    result: SearchResult,
) -> list[dict[str, Any]]:
    return [
        {
            "text": source.text,
            "score": source.score,
            "document": source.document,
            "chunk_id": source.chunk_id,
        }
        for source in result.sources
    ]


def display_sources(
    sources: list[dict[str, Any]],
) -> None:
    if not sources:
        return

    with st.expander(
        f"Sources ({len(sources)})",
        expanded=False,
    ):
        for index, source in enumerate(
            sources,
            start=1,
        ):
            document = source.get(
                "document",
                "Unknown document",
            )
            chunk_id = source.get(
                "chunk_id",
                "",
            )
            score = float(
                source.get(
                    "score",
                    0.0,
                )
            )
            text = source.get(
                "text",
                "",
            )

            st.markdown(f"**{index}. {document}**")

            metadata_parts = [f"Score: `{score:.4f}`"]

            if chunk_id:
                metadata_parts.append(f"Chunk: `{chunk_id}`")

            st.caption(" · ".join(metadata_parts))
            st.write(text)

            if index < len(sources):
                st.divider()


def display_answer_mode(answer_mode: str) -> None:
    if answer_mode == "general":
        st.caption(
            "General local-model answer · not grounded in uploaded documents",
        )
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

        with st.chat_message(role):
            st.markdown(content)

            if role == "assistant":
                display_answer_mode(answer_mode)
                display_sources(sources)


def render_sidebar(
    client: HomelabAPIClient,
) -> int:
    with st.sidebar:
        st.header("Homelab AI")
        st.caption("Local retrieval-augmented generation")

        st.divider()

        st.subheader("Conversations")
        if st.button(
            "New conversation",
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
            with st.expander("Conversation settings"):
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
            st.caption("Create a conversation to begin chatting.")

        st.divider()

        top_k = st.slider(
            label="Retrieved sources",
            min_value=1,
            max_value=20,
            value=DEFAULT_TOP_K,
            step=1,
            help=("Controls how many reranked passages are returned with each answer."),
        )

        st.divider()

        status = st.session_state.backend_status

        if status is None:
            st.error("Backend offline")
            st.caption(f"Start FastAPI on {API_URL}.")
        else:
            service = status.get(
                "service",
                "homelab-ai",
            )

            st.success("Backend online")
            st.caption(f"Service: `{service}`")

        if st.button(
            "Check backend",
            use_container_width=True,
        ):
            check_backend(client)
            st.rerun()

        st.divider()
        st.caption(f"API: `{API_URL}`")

    return int(top_k)


def render_chat_tab(
    client: HomelabAPIClient,
    top_k: int,
) -> None:
    st.subheader("Document chat")
    st.caption(
        "Ask questions about your documents. Conversations and citations are saved.",
    )

    display_chat_history()

    question = st.chat_input("Ask a question about your documents...")

    if question is None:
        return

    cleaned_question = question.strip()

    if not cleaned_question:
        st.warning("Enter a question before submitting.")
        return

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

    with st.chat_message("user"):
        st.markdown(cleaned_question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating an answer..."):
            try:
                conversation = client.send_conversation_message(
                    conversation_id=conversation_id,
                    content=cleaned_question,
                    top_k=top_k,
                )
            except ValueError as exc:
                error_message = str(exc)
                st.warning(error_message)

                add_message(
                    role="assistant",
                    content=error_message,
                )
                return

            except HomelabAPIError as exc:
                error_message = f"Request failed: {exc}"
                st.error(error_message)

                st.session_state.backend_status = None

                try:
                    load_conversation(
                        client.get_conversation(conversation_id),
                    )
                except HomelabAPIError:
                    pass
                return

        load_conversation(conversation)
        assistant_messages = [
            message for message in conversation.messages if message.role == "assistant"
        ]
        if assistant_messages:
            assistant = assistant_messages[-1]
            st.markdown(assistant.content)
            display_answer_mode(assistant.answer_mode)
            serialized_sources = [
                {
                    "text": source.text,
                    "score": source.score,
                    "document": source.document,
                    "chunk_id": source.chunk_id,
                }
                for source in assistant.sources
            ]
            display_sources(serialized_sources)


def render_documents_tab(
    client: HomelabAPIClient,
) -> None:
    st.subheader("Documents")
    st.caption(
        "Queue documents for durable background parsing, embedding, and indexing.",
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


def render_evaluation_tab(
    client: HomelabAPIClient,
) -> None:
    st.subheader("Retrieval evaluation")
    st.caption("Run the sample benchmark against the currently indexed corpus.")

    evaluation_top_k = st.slider(
        label="Evaluation retrieval depth",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
        key="evaluation_top_k",
        help=("Maximum number of retrieved documents evaluated for each benchmark question."),
    )

    if st.button(
        "Run evaluation",
        type="primary",
        use_container_width=True,
    ):
        with st.spinner("Running retrieval benchmark..."):
            try:
                result = client.evaluate(
                    top_k=int(evaluation_top_k),
                )
            except ValueError as exc:
                st.warning(str(exc))
            except HomelabAPIError as exc:
                st.error(f"Evaluation failed: {exc}")
                st.session_state.backend_status = None
            else:
                st.session_state.evaluation_result = result

    evaluation_result = st.session_state.evaluation_result

    if evaluation_result is None:
        st.info("Run the benchmark to calculate retrieval metrics.")
        return

    display_evaluation_result(evaluation_result)


def display_evaluation_result(
    result: EvaluationResult,
) -> None:
    metrics = result.metrics

    st.divider()

    first_row = st.columns(3)

    first_row[0].metric(
        "Hit@1",
        f"{metrics.hit_at_1:.1%}",
    )
    first_row[1].metric(
        "Hit@5",
        f"{metrics.hit_at_5:.1%}",
    )
    first_row[2].metric(
        "MRR",
        f"{metrics.mean_reciprocal_rank:.3f}",
    )

    second_row = st.columns(3)

    second_row[0].metric(
        "Precision@5",
        f"{metrics.precision_at_5:.1%}",
    )
    second_row[1].metric(
        "Recall@5",
        f"{metrics.recall_at_5:.1%}",
    )
    second_row[2].metric(
        "Questions",
        str(metrics.question_count),
    )

    st.caption(f"Top K: `{result.top_k}` · Elapsed time: `{result.elapsed_ms:.2f} ms`")

    st.divider()
    st.markdown("### Per-question results")

    rows = [
        {
            "Question": question.question,
            "Relevant documents": ", ".join(question.relevant_documents),
            "Retrieved documents": ", ".join(question.retrieved_documents),
            "Hit@1": question.hit_at_1,
            "Hit@5": question.hit_at_5,
            "Precision@5": (question.precision_at_5),
            "Recall@5": (question.recall_at_5),
            "Reciprocal rank": (question.reciprocal_rank),
        }
        for question in result.questions
    ]

    st.dataframe(
        rows,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Hit@1": (
                st.column_config.NumberColumn(
                    format="%.2f",
                )
            ),
            "Hit@5": (
                st.column_config.NumberColumn(
                    format="%.2f",
                )
            ),
            "Precision@5": (
                st.column_config.NumberColumn(
                    format="%.2f",
                )
            ),
            "Recall@5": (
                st.column_config.NumberColumn(
                    format="%.2f",
                )
            ),
            "Reciprocal rank": (
                st.column_config.NumberColumn(
                    format="%.3f",
                )
            ),
        },
    )


def main() -> None:
    st.set_page_config(
        page_title="Homelab AI",
        page_icon="🏠",
        layout="wide",
    )

    initialize_session_state()

    client = get_api_client()

    if st.session_state.backend_status is None:
        check_backend(client)

    top_k = render_sidebar(client)

    st.title("Homelab AI")

    (
        chat_tab,
        documents_tab,
        evaluation_tab,
    ) = st.tabs(
        [
            "Chat",
            "Documents",
            "Evaluation",
        ]
    )

    with chat_tab:
        render_chat_tab(
            client=client,
            top_k=top_k,
        )

    with documents_tab:
        render_documents_tab(
            client=client,
        )

    with evaluation_tab:
        render_evaluation_tab(
            client=client,
        )


if __name__ == "__main__":
    main()
