from typing import Any

import streamlit as st
from api import (
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


def get_api_client() -> HomelabAPIClient:
    return HomelabAPIClient()


def check_backend(client: HomelabAPIClient) -> None:
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
            chunk_id = source.get("chunk_id", "")
            score = float(source.get("score", 0.0))
            text = source.get("text", "")

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


def main() -> None:
    st.set_page_config(
        page_title="Homelab AI",
        page_icon="🏠",
        layout="centered",
    )

    initialize_session_state()

    client = get_api_client()

    if st.session_state.backend_status is None:
        check_backend(client)

    top_k = render_sidebar(client)

    st.title("Homelab AI")
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


if __name__ == "__main__":
    main()
