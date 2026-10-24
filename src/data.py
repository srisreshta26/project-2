
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO

import pandas as pd


DATA_URL = (
    "https://archive.ics.uci.edu/static/public/"
    "228/sms+spam+collection.zip"
)

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "SMSSpamCollection"


def load_dataset():
    """Download the public SMS dataset if needed, then load it."""

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        with urlopen(DATA_URL, timeout=30) as response:
            zip_bytes = response.read()

        with ZipFile(BytesIO(zip_bytes)) as archive:
            member = next(
                name for name in archive.namelist()
                if name.endswith("SMSSpamCollection")
            )
            DATA_PATH.write_bytes(archive.read(member))

    df = pd.read_csv(
        DATA_PATH,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="utf-8",
    )

    if df.empty or not {"label", "message"}.issubset(df.columns):
        raise ValueError("The dataset is empty or has an invalid format.")

    df = df.dropna(subset=["label", "message"])
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["message"] = df["message"].astype(str).str.strip()

    df = df[df["label"].isin(["ham", "spam"])]
    df = df[df["message"] != ""]

    if df.empty:
        raise ValueError("No valid messages were found in the dataset.")

    return df.reset_index(drop=True)
