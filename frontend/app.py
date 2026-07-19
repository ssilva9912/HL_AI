from typing import Any

import streamlit as st
from api import (
    EvaluationResult,
    HomelabAPIClient,
    HomelabAPIError,
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
            score = float(source.get("score", 0.0))
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

        with st.chat_message(role):
            st.markdown(content)

            if role == "assistant":
                display_sources(sources)


def render_sidebar(
    client: HomelabAPIClient,
) -> int:
    with st.sidebar:
        st.header("Homelab AI")
        st.caption("Local retrieval-augmented generation")

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

        if st.button(
            "Clear conversation",
            use_container_width=True,
        ):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.caption(f"API: `{API_URL}`")

    return int(top_k)


def render_chat_tab(
    client: HomelabAPIClient,
    top_k: int,
) -> None:
    st.subheader("Document chat")
    st.caption("Ask questions about documents indexed by your local RAG system.")

    display_chat_history()

    question = st.chat_input("Ask a question about your documents...")

    if question is None:
        return

    cleaned_question = question.strip()

    if not cleaned_question:
        st.warning("Enter a question before submitting.")
        return

    add_message(
        role="user",
        content=cleaned_question,
    )

    with st.chat_message("user"):
        st.markdown(cleaned_question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating an answer..."):
            try:
                result = client.search(
                    question=cleaned_question,
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

                add_message(
                    role="assistant",
                    content=error_message,
                )
                return

        st.markdown(result.answer)

        serialized_sources = serialize_sources(result)
        display_sources(serialized_sources)

        add_message(
            role="assistant",
            content=result.answer,
            sources=serialized_sources,
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
            "Precision@5": question.precision_at_5,
            "Recall@5": question.recall_at_5,
            "Reciprocal rank": (question.reciprocal_rank),
        }
        for question in result.questions
    ]

    st.dataframe(
        rows,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Hit@1": st.column_config.NumberColumn(format="%.2f"),
            "Hit@5": st.column_config.NumberColumn(format="%.2f"),
            "Precision@5": (st.column_config.NumberColumn(format="%.2f")),
            "Recall@5": (st.column_config.NumberColumn(format="%.2f")),
            "Reciprocal rank": (st.column_config.NumberColumn(format="%.3f")),
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

    chat_tab, evaluation_tab = st.tabs(
        [
            "Chat",
            "Evaluation",
        ]
    )

    with chat_tab:
        render_chat_tab(
            client=client,
            top_k=top_k,
        )

    with evaluation_tab:
        render_evaluation_tab(
            client=client,
        )


if __name__ == "__main__":
    main()
