
from src.data import load_dataset


def test_dataset_loads_successfully():
    df = load_dataset()

    assert not df.empty
    assert {"label", "message"}.issubset(df.columns)
    assert set(df["label"].unique()).issubset({"ham", "spam"})
    assert df["message"].str.strip().ne("").all()
