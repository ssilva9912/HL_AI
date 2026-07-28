import json
import logging
from collections.abc import Iterator
from time import perf_counter
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from backend.api.dependencies import get_rag_service
from backend.api.rag_service import HomelabRAGService
from backend.api.schemas import (
    ConversationAnswerResponse,
    ConversationCreateRequest,
    ConversationMessageRequest,
    ConversationMessageResponse,
    ConversationResponse,
    ConversationUpdateRequest,
    SearchMetadata,
    SourceResponse,
)
from backend.database import (
    Conversation,
    ConversationMessage,
    ConversationMessageRepository,
    ConversationRepository,
    MessageRole,
    get_database_session,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/conversations", tags=["conversations"])


def _source_responses(
    sources: list[dict[str, object]],
) -> list[SourceResponse]:
    responses: list[SourceResponse] = []
    for source in sources:
        raw_score = source.get("score", 0.0)
        score = float(raw_score) if isinstance(raw_score, (str, int, float)) else 0.0
        responses.append(
            SourceResponse(
                text=str(source.get("text", "")),
                score=score,
                document=(str(source["document"]) if source.get("document") is not None else None),
                chunk_id=(str(source["chunk_id"]) if source.get("chunk_id") is not None else None),
            ),
        )
    return responses


def _message_response(
    message: ConversationMessage,
) -> ConversationMessageResponse:
    return ConversationMessageResponse(
        id=message.id,
        conversation_id=message.conversation_id,
        role=message.role.value,
        content=message.content,
        answer_mode=message.answer_mode,
        sources=_source_responses(message.sources),
        created_at=message.created_at,
    )


def _conversation_response(
    conversation: Conversation,
    messages: list[ConversationMessage] | None = None,
) -> ConversationResponse:
    resolved_messages = messages or []
    return ConversationResponse(
        id=conversation.id,
        title=conversation.title,
        owner_key=conversation.owner_key,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        message_count=len(resolved_messages),
        messages=[_message_response(message) for message in resolved_messages],
    )


def _get_conversation_or_404(
    repository: ConversationRepository,
    conversation_id: UUID,
) -> Conversation:
    conversation = repository.get(conversation_id)
    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found.",
        )
    return conversation


@router.post(
    "",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_conversation(
    request: ConversationCreateRequest,
    session: Annotated[Session, Depends(get_database_session)],
) -> ConversationResponse:
    with session.begin():
        conversation = ConversationRepository(session).create(
            title=request.title,
        )
    return _conversation_response(conversation)


@router.get(
    "",
    response_model=list[ConversationResponse],
)
def list_conversations(
    session: Annotated[Session, Depends(get_database_session)],
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> list[ConversationResponse]:
    conversations = ConversationRepository(session).list_all(
        offset=offset,
        limit=limit,
    )
    messages = ConversationMessageRepository(session)
    return [
        _conversation_response(
            conversation,
            messages.list_for_conversation(conversation.id),
        )
        for conversation in conversations
    ]


@router.get(
    "/{conversation_id}",
    response_model=ConversationResponse,
)
def get_conversation(
    conversation_id: UUID,
    session: Annotated[Session, Depends(get_database_session)],
) -> ConversationResponse:
    conversation = _get_conversation_or_404(
        ConversationRepository(session),
        conversation_id,
    )
    messages = ConversationMessageRepository(session).list_for_conversation(
        conversation.id,
    )
    return _conversation_response(conversation, messages)


@router.patch(
    "/{conversation_id}",
    response_model=ConversationResponse,
)
def update_conversation(
    conversation_id: UUID,
    request: ConversationUpdateRequest,
    session: Annotated[Session, Depends(get_database_session)],
) -> ConversationResponse:
    with session.begin():
        repository = ConversationRepository(session)
        conversation = _get_conversation_or_404(repository, conversation_id)
        repository.rename(conversation, request.title)
    messages = ConversationMessageRepository(session).list_for_conversation(
        conversation.id,
    )
    return _conversation_response(conversation, messages)


@router.delete(
    "/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
def delete_conversation(
    conversation_id: UUID,
    session: Annotated[Session, Depends(get_database_session)],
) -> Response:
    with session.begin():
        repository = ConversationRepository(session)
        conversation = _get_conversation_or_404(repository, conversation_id)
        repository.delete(conversation)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/{conversation_id}/messages",
    response_model=ConversationAnswerResponse,
)
def add_conversation_message(
    conversation_id: UUID,
    request: ConversationMessageRequest,
    session: Annotated[Session, Depends(get_database_session)],
    rag_service: Annotated[HomelabRAGService, Depends(get_rag_service)],
) -> ConversationAnswerResponse:
    normalized_content = request.content.strip()
    if not normalized_content:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Message cannot be empty.",
        )

    conversations = ConversationRepository(session)
    message_repository = ConversationMessageRepository(session)
    conversation = _get_conversation_or_404(conversations, conversation_id)
    prior_messages = message_repository.list_for_conversation(conversation.id)
    history = [(message.role.value, message.content) for message in prior_messages]

    with session.begin_nested():
        user_message = message_repository.create(
            conversation=conversation,
            role=MessageRole.USER,
            content=normalized_content,
        )
        if conversation.title == "New conversation":
            conversations.rename(
                conversation,
                normalized_content[:80],
            )
    session.commit()

    started_at = perf_counter()
    try:
        result = rag_service.ask(
            question=normalized_content,
            top_k=request.top_k,
            history=history,
            hybrid=True,
        )
    except RuntimeError as exc:
        logger.exception("Conversation RAG configuration error")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected conversation RAG error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The message could not be processed.",
        ) from exc

    source_payloads: list[dict[str, object]] = [
        {
            "text": source.text,
            "score": source.score,
            "document": source.document,
            "chunk_id": source.chunk_id,
        }
        for source in result.sources
    ]
    with session.begin():
        conversation = _get_conversation_or_404(conversations, conversation_id)
        assistant_message = message_repository.create(
            conversation=conversation,
            role=MessageRole.ASSISTANT,
            content=result.answer,
            sources=source_payloads,
            answer_mode=result.answer_mode,
        )

    all_messages = message_repository.list_for_conversation(conversation.id)
    elapsed_ms = (perf_counter() - started_at) * 1_000
    return ConversationAnswerResponse(
        conversation=_conversation_response(conversation, all_messages),
        user_message=_message_response(user_message),
        assistant_message=_message_response(assistant_message),
        metadata=SearchMetadata(
            top_k=request.top_k,
            source_count=len(result.sources),
            elapsed_ms=round(elapsed_ms, 2),
        ),
    )


@router.post(
    "/{conversation_id}/messages/stream",
    response_class=StreamingResponse,
)
def stream_conversation_message(
    conversation_id: UUID,
    request: ConversationMessageRequest,
    session: Annotated[Session, Depends(get_database_session)],
    rag_service: Annotated[HomelabRAGService, Depends(get_rag_service)],
) -> StreamingResponse:
    normalized_content = request.content.strip()
    if not normalized_content:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Message cannot be empty.",
        )

    conversations = ConversationRepository(session)
    messages = ConversationMessageRepository(session)
    conversation = _get_conversation_or_404(conversations, conversation_id)
    prior_messages = messages.list_for_conversation(conversation.id)
    history = [(message.role.value, message.content) for message in prior_messages]

    with session.begin_nested():
        messages.create(
            conversation=conversation,
            role=MessageRole.USER,
            content=normalized_content,
        )
        if conversation.title == "New conversation":
            conversations.rename(conversation, normalized_content[:80])
    session.commit()

    def encode_event(event: dict[str, object]) -> str:
        return json.dumps(event, separators=(",", ":"), default=str) + "\n"

    def generate_events() -> Iterator[str]:
        started_at = perf_counter()
        try:
            result = rag_service.stream_answer(
                question=normalized_content,
                top_k=request.top_k,
                history=history,
                hybrid=True,
            )
            source_payloads: list[dict[str, object]] = [
                {
                    "text": source.text,
                    "score": source.score,
                    "document": source.document,
                    "chunk_id": source.chunk_id,
                }
                for source in result.sources
            ]
            yield encode_event(
                {
                    "type": "start",
                    "answer_mode": result.answer_mode,
                    "sources": source_payloads,
                }
            )

            chunks: list[str] = []
            for chunk in result.chunks:
                chunks.append(chunk)
                yield encode_event({"type": "token", "content": chunk})

            answer = "".join(chunks).strip()
            if not answer:
                raise RuntimeError("The model returned an empty response.")

            with session.begin():
                current_conversation = _get_conversation_or_404(
                    conversations,
                    conversation_id,
                )
                assistant = messages.create(
                    conversation=current_conversation,
                    role=MessageRole.ASSISTANT,
                    content=answer,
                    sources=source_payloads,
                    answer_mode=result.answer_mode,
                )

            yield encode_event(
                {
                    "type": "complete",
                    "assistant_message": _message_response(assistant).model_dump(mode="json"),
                    "elapsed_ms": round((perf_counter() - started_at) * 1_000, 2),
                }
            )
        except GeneratorExit:
            raise
        except Exception as exc:
            session.rollback()
            logger.exception("Streaming conversation generation failed")
            yield encode_event(
                {
                    "type": "error",
                    "detail": str(exc) or "The message could not be processed.",
                }
            )

    return StreamingResponse(
        generate_events(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
