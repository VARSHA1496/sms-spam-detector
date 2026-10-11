"""
data_loader.py
--------------
Loads the real "SMS Spam Collection" dataset (UCI Machine Learning Repository,
5,574 English SMS messages labelled "ham" = not spam or "spam").

The first time you run the project the dataset is downloaded automatically
into the  data/  folder. If you have no internet, put the file there yourself
(see README.md) - either:
    data/SMSSpamCollection   (original UCI file, tab-separated), or
    data/spam.csv            (the Kaggle version of the same dataset)
"""
import csv
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TSV_FILE = DATA_DIR / "SMSSpamCollection"   # UCI format: label <TAB> message
KAGGLE_FILE = DATA_DIR / "spam.csv"         # Kaggle format: v1,v2 columns

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
MIRROR_URL = (
    "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/"
    "master/data/sms.tsv"
)

MANUAL_HELP = (
    "Could not download the dataset automatically.\n"
    "Please download it by hand:\n"
    "  1. Open https://archive.ics.uci.edu/dataset/228/sms+spam+collection\n"
    "  2. Click 'Download', unzip the file, and copy the file named\n"
    "     'SMSSpamCollection' into the  data  folder of this project.\n"
    "  (Or put the Kaggle file 'spam.csv' into the  data  folder.)"
)


def _download_bytes(url, timeout=30):
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def download_dataset():
    """Download the dataset into data/SMSSpamCollection (tries 2 sources)."""
    DATA_DIR.mkdir(exist_ok=True)
    problems = []

    # Source 1: official UCI zip file
    try:
        raw = _download_bytes(UCI_ZIP_URL)
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            member = next(
                name for name in archive.namelist()
                if name.lower().endswith("smsspamcollection")
            )
            TSV_FILE.write_bytes(archive.read(member))
        _read_tsv(TSV_FILE)          # make sure the file is readable
        return
    except Exception as error:       # noqa: BLE001 - we try the next source
        problems.append(f"UCI: {error}")
        TSV_FILE.unlink(missing_ok=True)

    # Source 2: a public GitHub copy of the same data (same tab-separated format)
    try:
        TSV_FILE.write_bytes(_download_bytes(MIRROR_URL))
        _read_tsv(TSV_FILE)
        return
    except Exception as error:       # noqa: BLE001
        problems.append(f"Mirror: {error}")
        TSV_FILE.unlink(missing_ok=True)

    raise RuntimeError(MANUAL_HELP + "\n\nDetails: " + " | ".join(problems))


def _decode(path):
    data = path.read_bytes()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")   # the £ sign appears in many messages


def _read_tsv(path):
    return pd.read_csv(
        io.StringIO(_decode(path)),
        sep="\t",
        header=None,
        names=["label", "message"],
        quoting=csv.QUOTE_NONE,          # messages contain quote characters
        dtype=str,
        on_bad_lines="skip",
    )


def _read_kaggle_csv(path):
    df = pd.read_csv(io.StringIO(_decode(path)), usecols=[0, 1], dtype=str)
    df.columns = ["label", "message"]
    return df


def load_dataset():
    """Return a clean DataFrame with columns: label ('ham'/'spam'), message."""
    if TSV_FILE.exists():
        df = _read_tsv(TSV_FILE)
    elif KAGGLE_FILE.exists():
        df = _read_kaggle_csv(KAGGLE_FILE)
    else:
        download_dataset()
        df = _read_tsv(TSV_FILE)

    df = df.dropna()
    df["label"] = df["label"].str.strip().str.lower()
    df["message"] = df["message"].str.strip()
    df = df[df["label"].isin(["ham", "spam"]) & (df["message"] != "")]
    df = df.reset_index(drop=True)
    if df.empty:
        raise RuntimeError("The dataset file was empty or in an unknown format.\n" + MANUAL_HELP)
    return df
