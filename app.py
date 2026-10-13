import streamlit as st

from src.config import config
from src.models.schemas import DocumentPage, SourceReference
from src.services.document_service import (
    calculate_content_hash,
    chunk_document_pages,
    extract_pdf_pages,
)
from src.services.export_service import (
    export_answer_to_markdown,
    export_quiz_to_markdown,
    export_summary_to_markdown,
)
from src.services.index_service import build_workspace_index
from src.services.rag_service import answer_document_query
from src.services.study_service import generate_quiz, generate_study_notes
from src.services.summary_service import summarize_full_document
from src.utils.errors import RAGError

# ==============================================================================
# PAGE CONFIGURATION & RESPONSIVE STYLING
# ==============================================================================
st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Mobile-friendly CSS
st.markdown(
    """
    <style>
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1rem 0.5rem;
        }
        .stButton button {
            width: 100% !important;
        }
    }
    .citation-badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        margin: 0.2rem;
        border-radius: 4px;
        background-color: rgba(128, 128, 128, 0.15);
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📚 AI Document Assistant")
st.caption("Upload PDFs, ask grounded questions, summarize full documents, and generate study assets.")

# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "pdf_pages" not in st.session_state:
    st.session_state.pdf_pages = []

# ==============================================================================
# SIDEBAR: DOCUMENT LIBRARY & WORKSPACE MANAGEMENT
# ==============================================================================
with st.sidebar:
    st.header("📄 Document Workspace")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help=f"Maximum allowed size per file: {config.MAX_UPLOAD_MB} MB",
    )

    if st.button("Process PDFs", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload at least one PDF file first.")
        else:
            try:
                progress_bar = st.progress(0.0)
                status_text = st.empty()

                status_text.info("Validating and extracting text from PDFs...")
                all_pages: list[DocumentPage] = []
                seen_hashes: set[str] = set()
                low_text_pages = 0

                for i, file in enumerate(uploaded_files, start=1):
                    file_bytes = file.getvalue()
                    content_hash = calculate_content_hash(file_bytes)

                    if content_hash in seen_hashes:
                        st.info(f"Skipping duplicate upload: {file.name}")
                        continue
                    seen_hashes.add(content_hash)

                    pages = extract_pdf_pages(file_bytes, filename=file.name)
                    all_pages.extend(pages)
                    low_text_pages += sum(1 for p in pages if p.is_low_text)

                    progress_bar.progress(0.4 * (i / len(uploaded_files)))

                if not all_pages or all(not p.text.strip() for p in all_pages):
                    raise ValueError(
                        "No extractable text was found in the uploaded documents. "
                        "The files may be scanned images or password protected."
                    )

                status_text.info("Chunking document pages with metadata...")
                progress_bar.progress(0.6)
                chunks = chunk_document_pages(all_pages)

                status_text.info("Building isolated vector index...")
                progress_bar.progress(0.8)
                workspace_index = build_workspace_index(chunks)

                # Persist in session state
                st.session_state.vector_store = workspace_index
                st.session_state.pdf_pages = all_pages
                st.session_state.processed_files = [f.name for f in uploaded_files]
                st.session_state.page_count = len(all_pages)
                st.session_state.chunk_count = len(chunks)
                st.session_state.low_text_count = low_text_pages
                st.session_state.messages = []

                progress_bar.progress(1.0)
                status_text.success("Workspace ready!")
                st.rerun()

            except RAGError as rag_err:
                st.error(rag_err.user_message)
            except Exception as err:
                st.error(f"Could not process PDFs: {err}")

    # Display active workspace statistics
    if "vector_store" in st.session_state:
        st.divider()
        st.success("✅ Knowledge Base Active")
        st.write(f"**Indexed Files:** {len(st.session_state.processed_files)}")
        st.write(f"**Total Pages:** {st.session_state.page_count}")
        st.write(f"**Indexed Chunks:** {st.session_state.chunk_count}")

        if st.session_state.get("low_text_count", 0) > 0:
            st.warning(
                f"⚠️ {st.session_state.low_text_count} page(s) contain little or no text. "
                "These pages may be scanned images."
            )

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        with col2:
            if st.button("Reset All", use_container_width=True):
                for k in list(st.session_state.keys()):
                    del st.session_state[k]
                st.rerun()

    st.divider()
    st.caption("Powered by Streamlit, HuggingFace & Gemini")

# ==============================================================================
# MAIN WORKSPACE: MODE SELECTOR & MESSAGES
# ==============================================================================
mode = st.selectbox(
    "Choose a tool",
    [
        "Ask questions",
        "Summarize documents",
        "Generate study notes",
        "Generate quiz",
    ],
)

# Render conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Referenced Source Pages"):
                for src in msg["sources"]:
                    st.markdown(f"**{src.get('source')}** — Page {src.get('page')}")
                    if src.get("snippet"):
                        st.caption(f"> {src.get('snippet')}")

# User input prompts
prompts = {
    "Ask questions": "Ask a question about your uploaded PDFs...",
    "Summarize documents": "Type 'summarize' or specify sections/focus...",
    "Generate study notes": "Which topic should the study notes cover?",
    "Generate quiz": "What topic should the quiz cover?",
}

user_input = st.chat_input(prompts[mode])

# ==============================================================================
# WORKFLOW EXECUTION DISPATCHER
# ==============================================================================
if user_input:
    if "vector_store" not in st.session_state:
        st.warning("Please upload and process at least one PDF document first.")
        st.stop()

    # Append user question to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            # 1. Ask questions mode
            if mode == "Ask questions":
                with st.spinner("Searching evidence and generating grounded response..."):
                    result = answer_document_query(
                        vector_store=st.session_state.vector_store,
                        query=user_input,
                    )

                st.markdown(result.answer)

                source_dicts = []
                if result.sources:
                    with st.expander("📄 Referenced Source Pages"):
                        for s in result.sources:
                            st.markdown(f"**{s.source}** — Page {s.page}")
                            if s.snippet:
                                st.caption(f"> {s.snippet}")
                            source_dicts.append({"source": s.source, "page": s.page, "snippet": s.snippet})

                # Export download button
                download_content = export_answer_to_markdown(result)
                st.download_button(
                    label="📥 Download Answer (.md)",
                    data=download_content,
                    file_name="answer_result.md",
                    mime="text/markdown",
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result.answer,
                        "sources": source_dicts,
                    }
                )

            # 2. Summarize documents mode
            elif mode == "Summarize documents":
                progress_bar = st.progress(0.0)
                status_text = st.empty()

                def update_summary_progress(fraction: float, message: str):
                    progress_bar.progress(min(fraction, 1.0))
                    status_text.info(message)

                with st.spinner("Executing full-document hierarchical Map-Reduce summarization..."):
                    summary_res = summarize_full_document(
                        pages=st.session_state.pdf_pages,
                        progress_callback=update_summary_progress,
                    )

                progress_bar.progress(1.0)
                status_text.empty()

                if summary_res.status == "partial":
                    st.warning("⚠️ Full-document summary is partial: Some sections could not be summarized.")

                st.markdown(summary_res.summary)

                source_dicts = []
                if summary_res.sources:
                    with st.expander("📄 Covered Pages"):
                        for s in summary_res.sources:
                            st.write(f"- {s.source} — Page {s.page}")
                            source_dicts.append({"source": s.source, "page": s.page})

                download_content = export_summary_to_markdown(summary_res)
                st.download_button(
                    label="📥 Download Full Summary (.md)",
                    data=download_content,
                    file_name="document_summary.md",
                    mime="text/markdown",
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": summary_res.summary,
                        "sources": source_dicts,
                    }
                )

            # 3. Generate study notes mode
            elif mode == "Generate study notes":
                with st.spinner("Extracting relevant context and drafting study notes..."):
                    # Retrieve context for topic
                    rag_result = answer_document_query(
                        vector_store=st.session_state.vector_store,
                        query=user_input,
                    )
                    context_text = rag_result.answer

                    notes = generate_study_notes(
                        topic=user_input,
                        context=context_text,
                    )

                st.markdown(notes)

                st.download_button(
                    label="📥 Download Study Notes (.md)",
                    data=notes,
                    file_name="study_notes.md",
                    mime="text/markdown",
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": notes,
                        "sources": [{"source": s.source, "page": s.page} for s in rag_result.sources],
                    }
                )

            # 4. Generate quiz mode
            elif mode == "Generate quiz":
                with st.spinner("Formulating multiple-choice questions with answer keys..."):
                    rag_result = answer_document_query(
                        vector_store=st.session_state.vector_store,
                        query=user_input,
                    )
                    quiz_res = generate_quiz(
                        topic=user_input,
                        context=rag_result.answer,
                    )

                if quiz_res.status == "failed":
                    st.error("Failed to generate a valid quiz. Please try a different topic.")
                else:
                    st.subheader(f"Quiz: {user_input}")
                    for idx, item in enumerate(quiz_res.items, start=1):
                        st.markdown(f"**Q{idx}: {item.question}**")
                        for o_idx, opt in enumerate(item.options):
                            letter = chr(ord("A") + o_idx)
                            st.write(f"- **{letter})** {opt}")
                        st.write("")

                    with st.expander("🔑 Reveal Answer Key"):
                        for idx, item in enumerate(quiz_res.items, start=1):
                            letter = chr(ord("A") + item.correct_answer_index)
                            st.markdown(f"**Q{idx}: {letter}) {item.options[item.correct_answer_index]}**")
                            if item.explanation:
                                st.caption(f"*Why:* {item.explanation}")

                    download_content = export_quiz_to_markdown(quiz_res)
                    st.download_button(
                        label="📥 Download Quiz Sheet (.md)",
                        data=download_content,
                        file_name="quiz_sheet.md",
                        mime="text/markdown",
                    )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": f"Generated {len(quiz_res.items)} questions on '{user_input}'.",
                            "sources": [{"source": s.source, "page": s.page} for s in rag_result.sources],
                        }
                    )

        except RAGError as rag_err:
            st.error(rag_err.user_message)
        except Exception as err:
            st.error(f"Request failed: {err}")
