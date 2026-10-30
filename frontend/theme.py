import streamlit as st

USER_AVATAR = ":material/person:"
ASSISTANT_AVATAR = ":material/auto_awesome:"

TAGLINE = "Your private assistant. Answers come from your own documents, on your own machine."

# (label, icon, page) shortcuts shown on an empty chat; every one works whatever is indexed.
QUICK_ACTIONS = (
    ("Add documents", ":material/upload_file:", "documents"),
    ("Fill out a form", ":material/edit_document:", "forms"),
    ("See live metrics", ":material/monitoring:", "metrics"),
)

_STYLES = """
<style>
.block-container { max-width: 1080px; padding-top: 4.6rem; }
footer { visibility: hidden; }

/* header */
.hl-hero { display: flex; align-items: center; justify-content: space-between;
           gap: 1rem; margin: 0 0 1.4rem 0; flex-wrap: wrap; }
.hl-brand { display: flex; align-items: center; gap: 0.95rem; }
.hl-logo { width: 48px; height: 48px; border-radius: 15px; display: grid; place-items: center;
           background: linear-gradient(135deg, #6366f1 0%, #22d3ee 100%); color: #fff;
           font-weight: 700; letter-spacing: 0.5px;
           box-shadow: 0 8px 22px rgba(99, 102, 241, 0.35); }
.hl-title { font-size: 1.8rem; font-weight: 700; line-height: 1.1; margin: 0; }
.hl-tagline { opacity: 0.7; font-size: 0.95rem; margin: 0.2rem 0 0 0; }

/* status pill */
.hl-pill { display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.3rem 0.8rem;
           border-radius: 999px; font-size: 0.8rem;
           border: 1px solid rgba(128, 128, 128, 0.3); background: rgba(128, 128, 128, 0.06); }
.hl-dot { width: 8px; height: 8px; border-radius: 50%; background: #34d399;
          box-shadow: 0 0 0 3px rgba(52, 211, 153, 0.22); }
.hl-pill.off .hl-dot { background: #f87171; box-shadow: 0 0 0 3px rgba(248, 113, 113, 0.22); }

/* tabs */
button[data-baseweb="tab"] { padding: 0.65rem 1.05rem; font-weight: 600; }

/* chat */
[data-testid="stChatMessage"] { border: 1px solid rgba(128, 128, 128, 0.2); border-radius: 18px;
                                padding: 1rem 1.15rem; margin-bottom: 0.8rem;
                                background: rgba(128, 128, 128, 0.05); }
[data-testid="stChatInput"] { border-radius: 16px; }

/* buttons */
.stButton > button, .stDownloadButton > button, .stLinkButton > a {
    border-radius: 12px; transition: transform 0.12s ease, box-shadow 0.12s ease; }
.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-1px); box-shadow: 0 6px 16px rgba(0, 0, 0, 0.18); }

/* welcome card */
.hl-welcome { border: 1px solid rgba(128, 128, 128, 0.22); border-radius: 22px;
              padding: 1.6rem 1.8rem; margin: 0.4rem 0 1rem 0;
              background: linear-gradient(135deg, rgba(99, 102, 241, 0.12),
                                          rgba(34, 211, 238, 0.08)); }
.hl-welcome h3 { margin: 0 0 0.35rem 0; font-size: 1.35rem; }
.hl-welcome p { margin: 0; opacity: 0.8; }

/* sidebar */
section[data-testid="stSidebar"] { border-right: 1px solid rgba(128, 128, 128, 0.16); }
.hl-side-brand { display: flex; align-items: center; gap: 0.65rem; margin: 0.2rem 0 0.9rem 0; }
.hl-side-brand .hl-logo { width: 34px; height: 34px; border-radius: 11px; font-size: 0.8rem; }
.hl-side-brand span { font-weight: 700; font-size: 1.1rem; }
.hl-section { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.09em;
              text-transform: uppercase; opacity: 0.55; margin: 0.4rem 0 0.35rem 0; }
</style>
"""


def inject_styles() -> None:
    st.markdown(_STYLES, unsafe_allow_html=True)


def status_pill(online: bool) -> str:
    label = "Online" if online else "Offline"
    return (
        f'<span class="hl-pill {"" if online else "off"}">'
        f'<span class="hl-dot"></span>{label}</span>'
    )


def header_html(online: bool) -> str:
    return (
        '<div class="hl-hero"><div class="hl-brand"><div class="hl-logo">AI</div><div>'
        '<p class="hl-title">Homelab AI</p>'
        f'<p class="hl-tagline">{TAGLINE}</p></div></div>'
        f"{status_pill(online)}</div>"
    )


def welcome_html() -> str:
    return (
        '<div class="hl-welcome"><h3>Hi there, how can I help?</h3>'
        "<p>Ask me about something in your documents, like <em>“What does my contract say "
        "about notice periods?”</em> I'll show you exactly which pages and files each answer "
        "came from, so you can check it yourself.</p></div>"
    )


def section_label(text: str) -> str:
    return f'<div class="hl-section">{text}</div>'


def sidebar_brand_html() -> str:
    return '<div class="hl-side-brand"><div class="hl-logo">AI</div><span>Homelab AI</span></div>'
