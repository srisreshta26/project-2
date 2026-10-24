
from src.data import load_dataset


def main():
    df = load_dataset()
    print(f"Dataset loaded successfully: {len(df)} messages")
    print("\nLabel counts:")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()
