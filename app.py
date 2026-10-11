"""
app.py - Streamlit web interface for the SMS Spam Detection System.

Run with:   streamlit run app.py
"""
import html
import re

import pandas as pd
import streamlit as st

import spam_model as sm

st.set_page_config(page_title="SMS Spam Checker", page_icon="📱", layout="centered")

# ---------------------------------------------------------------- styling ---
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {
  --ink: #18212B; --muted: #566170; --line: #D5DBE3; --paper: #F2F4F7;
  --spam: #B3261E; --spam-bg: #FBE7E4; --ham: #17694D; --ham-bg: #E0F2EA;
  --accent: #2F4C8C;
}
html, body, .stApp, .stApp p, .stApp label, .stApp li, .stApp button {
  font-family: 'IBM Plex Sans', system-ui, 'Segoe UI', sans-serif;
}
.stApp { background: var(--paper); color: var(--ink); }
.block-container { max-width: 760px; padding-top: 2.4rem; }
#MainMenu, footer { visibility: hidden; }

.page-title {
  font-family: 'Bricolage Grotesque', 'Segoe UI', sans-serif;
  font-weight: 700; font-size: 2.5rem; letter-spacing: -0.02em;
  line-height: 1.1; margin: 0 0 .5rem 0; color: var(--ink);
}
.page-sub { color: var(--muted); font-size: 1.05rem; line-height: 1.55; margin-bottom: 1.4rem; max-width: 60ch; }

.stTextArea textarea {
  font-size: 1.05rem; line-height: 1.5; border-radius: 14px;
  border: 1.5px solid var(--line); background: #FFFFFF; color: var(--ink);
}
.stTextArea textarea:focus { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(47,76,140,.18); }

.stButton > button { border-radius: 10px; font-weight: 500; border: 1.5px solid var(--line); background: #FFFFFF; color: var(--ink); }
.stButton > button:hover { border-color: var(--accent); color: var(--accent); }
.stButton > button[kind="primary"] { background: var(--accent); border-color: var(--accent); color: #FFFFFF; font-weight: 600; }
.stButton > button[kind="primary"]:hover { background: #243B6F; color: #FFFFFF; }

.verdict { border-radius: 18px; padding: 1.3rem 1.5rem 1.4rem; margin-top: 1.2rem; border: 1.5px solid; }
.verdict.spam { background: var(--spam-bg); border-color: var(--spam); }
.verdict.ham  { background: var(--ham-bg);  border-color: var(--ham); }
.verdict-word { font-family: 'Bricolage Grotesque', 'Segoe UI', sans-serif; font-size: 2.6rem; font-weight: 700; letter-spacing: -0.02em; line-height: 1; }
.verdict.spam .verdict-word { color: var(--spam); }
.verdict.ham .verdict-word { color: var(--ham); }
.verdict-prob { margin-top: .35rem; font-size: 1rem; color: var(--ink); }
.bar { height: 10px; border-radius: 6px; background: rgba(24,33,43,.12); margin: .9rem 0 1.1rem; overflow: hidden; }
.bar > div { height: 100%; border-radius: 6px; }
.verdict.spam .bar > div { background: var(--spam); }
.verdict.ham .bar > div { background: var(--ham); }
.bubble { background: #FFFFFF; border-radius: 16px 16px 16px 4px; padding: .9rem 1.1rem; line-height: 1.65; font-size: 1.02rem; color: var(--ink); border: 1px solid var(--line); word-wrap: break-word; }
mark.sp { background: #F6C3BD; color: #7D1913; padding: 0 .2em; border-radius: 4px; }
mark.hm { background: #BFE5D3; color: #0F4A35; padding: 0 .2em; border-radius: 4px; }
.legend { margin-top: .8rem; color: var(--muted); font-size: .92rem; }
.note { color: var(--muted); font-size: .95rem; margin-top: .9rem; }
.stTabs [data-baseweb="tab"] { font-weight: 500; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ------------------------------------------------------------ model setup ---
EXAMPLES = {
    "Prize text": "Congratulations! You've won a FREE £500 gift card. Text CLAIM to 80082 now to collect your prize. Reply STOP to opt out.",
    "Subscription offer": "URGENT! You have won a 1 week FREE membership in our £100,000 prize jackpot! Txt the word CLAIM to No: 81010 now. T&C www.dbuk.net",
    "Lunch plans": "Hey, are we still meeting for lunch at 1? Let me know if you want me to book the table.",
    "Lecture notes": "Can you send me the notes from today's lecture? I missed the last half hour.",
}


def ensure_model():
    """Train the model automatically the very first time (needs internet once)."""
    if sm.model_exists():
        return
    with st.spinner("First start: downloading the dataset and training the model (about 20 seconds) ..."):
        import train_model
        train_model.train_and_save(verbose=False)


@st.cache_resource(show_spinner=False)
def get_model():
    return sm.load_model()


@st.cache_data(show_spinner=False)
def get_metrics():
    return sm.load_metrics()


try:
    ensure_model()
    model = get_model()
    metrics = get_metrics()
except Exception as error:  # noqa: BLE001 - show a readable message instead of a crash
    st.error("The model could not be prepared.")
    st.code(str(error))
    st.info("Open a terminal in the project folder and run:  python train_model.py")
    st.stop()


# ---------------------------------------------------------------- helpers ---
def highlight(text, spam_words, ham_words):
    """Return the message as safe HTML, colouring the words that drove the decision."""
    pieces = []
    previous, before_previous = "", ""
    for part in re.findall(r"\w+|\W+", text):
        safe = html.escape(part).replace("$", "&#36;").replace("\n", "<br>")
        lowered = part.lower()
        if previous in ("'", "\u2019") and before_previous[-1:].isalnum():
            pieces.append(safe)                       # skip pieces like the "ve" in "you've"
        elif lowered in spam_words:
            pieces.append(f'<mark class="sp">{safe}</mark>')
        elif lowered in ham_words:
            pieces.append(f'<mark class="hm">{safe}</mark>')
        else:
            pieces.append(safe)
        before_previous, previous = previous, part
    return "".join(pieces)


def use_example(text):
    st.session_state.sms_text = text
    st.session_state.run_check = True


st.session_state.setdefault("sms_text", "")

# ------------------------------------------------------------------ header ---
st.markdown('<div class="page-title">SMS spam checker</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="page-sub">Paste a text message and find out whether it looks like spam. '
    f'The model learned from {metrics["n_total"]:,} real SMS messages.</div>',
    unsafe_allow_html=True,
)

tab_check, tab_perf, tab_how = st.tabs(["Check a message", "Model performance", "How it works"])

# --------------------------------------------------------------- tab 1: check
with tab_check:
    st.text_area(
        "SMS message",
        key="sms_text",
        height=130,
        placeholder="Type or paste an SMS message here ...",
        label_visibility="collapsed",
    )

    st.caption("Or try an example")
    columns = st.columns(len(EXAMPLES))
    for column, (name, text) in zip(columns, EXAMPLES.items()):
        column.button(name, key=f"ex_{name}", on_click=use_example, args=(text,))

    clicked = st.button("Check message", type="primary")
    auto_run = st.session_state.pop("run_check", False)

    if clicked or auto_run:
        message = st.session_state.sms_text.strip()
        if not message:
            st.warning("Enter a message first, then select Check message.")
        else:
            label, spam_p = sm.predict_message(model, message)
            spam_words, ham_words = sm.explain_message(model, message)
            is_spam = label == "spam"
            shown_p = spam_p if is_spam else 1 - spam_p
            body = highlight(
                message,
                {w for w, _ in spam_words},
                {w for w, _ in ham_words},
            )
            st.markdown(
                f'<div class="verdict {"spam" if is_spam else "ham"}">'
                f'<div class="verdict-word">{"Spam" if is_spam else "Not spam"}</div>'
                f'<div class="verdict-prob">{shown_p * 100:.1f}% confidence'
                f' &nbsp;(spam probability {spam_p * 100:.1f}%)</div>'
                f'<div class="bar"><div style="width:{shown_p * 100:.1f}%"></div></div>'
                f'<div class="bubble">{body}</div>'
                f'<div class="legend">'
                f'<mark class="sp">red</mark> words pushed the decision towards spam, '
                f'<mark class="hm">green</mark> words pushed it towards not spam.</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if 0.35 <= spam_p <= 0.65:
                st.markdown(
                    '<div class="note">The model is close to undecided on this message, '
                    'so treat the result with caution.</div>',
                    unsafe_allow_html=True,
                )
            if spam_words or ham_words:
                left, right = st.columns(2)
                with left:
                    st.markdown("**Spam signals**")
                    st.write(", ".join(w for w, _ in spam_words) or "none found")
                with right:
                    st.markdown("**Normal-message signals**")
                    st.write(", ".join(w for w, _ in ham_words) or "none found")
            else:
                st.markdown(
                    '<div class="note">None of the words in this message appear in the '
                    'model\'s vocabulary, so the result mostly reflects how common spam is '
                    'in the training data.</div>',
                    unsafe_allow_html=True,
                )

# ------------------------------------------------------------ tab 2: metrics
with tab_perf:
    st.markdown(
        f"Measured on **{metrics['n_test']:,} messages the model never saw during training** "
        f"({metrics['n_train']:,} were used for training)."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
    c2.metric("Precision", f"{metrics['precision'] * 100:.2f}%")
    c3.metric("Recall", f"{metrics['recall'] * 100:.2f}%")
    c4.metric("F1-score", f"{metrics['f1'] * 100:.2f}%")
    st.caption(
        "Accuracy: share of all messages classified correctly. "
        "Precision: of the messages flagged as spam, how many really were spam. "
        "Recall: of all real spam, how much was caught. "
        "F1-score: balance of precision and recall."
    )
    st.markdown(
        f"5-fold cross-validation accuracy on the training data: "
        f"**{metrics['cv_mean'] * 100:.2f}%** (± {metrics['cv_std'] * 100:.2f})"
    )

    st.markdown("**Confusion matrix**")
    cm = metrics["confusion"]
    st.dataframe(
        pd.DataFrame(
            [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
            index=["Actually not spam", "Actually spam"],
            columns=["Predicted not spam", "Predicted spam"],
        )
    )

    st.markdown("**Dataset**")
    st.write(
        f"{metrics['n_raw']:,} messages in the original file, "
        f"{metrics['n_total']:,} after removing exact duplicates: "
        f"{metrics['n_ham']:,} not spam and {metrics['n_spam']:,} spam. "
        f"The model uses {metrics['n_features']:,} word and word-pair features."
    )

    left, right = st.columns(2)
    with left:
        st.markdown("**Most typical spam words**")
        st.bar_chart(pd.DataFrame(metrics["top_spam_words"], columns=["word", "weight"]).set_index("word"))
    with right:
        st.markdown("**Most typical normal words**")
        st.bar_chart(pd.DataFrame(metrics["top_ham_words"], columns=["word", "weight"]).set_index("word"))

# ---------------------------------------------------------------- tab 3: how
with tab_how:
    st.markdown(
        """
The system turns a message into numbers, then asks which class those numbers fit better.

1. **Clean and split.** The real SMS Spam Collection (UCI) is loaded, duplicates are removed, and 80% of the messages are kept for training and 20% for testing.
2. **TF-IDF vectorization.** Each message becomes a list of word weights. A word gets a high weight when it is frequent in that message but rare across all messages, so "claim" counts for more than "the".
3. **Multinomial Naive Bayes.** The model learns how likely each word is in spam and in normal messages.
4. **Prediction.** For a new message it combines those word probabilities with Bayes' rule and picks the more likely class.
        """
    )
    st.latex(r"P(\text{spam}\mid \text{message}) \;\propto\; P(\text{spam}) \prod_{i} P(w_i \mid \text{spam})")
    st.markdown(
        "The model is called *naive* because it treats every word as independent of the others. "
        "That is not strictly true, but it works well for short text and trains in under a second."
    )
    st.markdown(
        "**Limits.** The data is English SMS from around 2012, so newer scam styles "
        "(for example delivery-fee links) may be missed, and the model does not understand meaning, only word patterns."
    )
