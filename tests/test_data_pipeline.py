import json
from pathlib import Path

from src.data.generator import SFTDataGenerator, dump_samples_jsonl, load_samples_jsonl


def test_sft_generate_and_jsonl_roundtrip(tmp_path: Path) -> None:
    generator = SFTDataGenerator(seed=123)
    prompts = ["查一下茅台", "查一下腾讯"]
    samples = generator.generate_batch(prompts)

    out = tmp_path / "samples.jsonl"
    dump_samples_jsonl(samples, str(out))

    loaded = load_samples_jsonl(str(out))
    assert len(loaded) == 2
    assert loaded[0].messages


def test_sft_save_to_parquet(tmp_path: Path) -> None:
    pd = __import__("pandas")
    assert pd is not None

    generator = SFTDataGenerator(seed=1)
    samples = generator.generate_batch(["a", "b", "c", "d"]) 
    train_path, val_path = generator.save_to_parquet(samples, str(tmp_path), split=0.75)

    assert Path(train_path).exists()
    assert Path(val_path).exists()

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    assert set(train_df.columns) == {"messages", "data_source"}
    assert set(val_df.columns) == {"messages", "data_source"}
    assert json.loads(train_df.iloc[0]["messages"])
