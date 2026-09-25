import pandas as pd

from sdg_classifier.data import SDG_COLUMNS, load_abstracts, one_label_per_row, split


def make_csv(tmp_path, rows):
    df = pd.DataFrame(rows).reindex(columns=["doi", "abstract", *SDG_COLUMNS])
    path = tmp_path / "abstracts.csv"
    df.to_csv(path, index=False)
    return path


def test_load_abstracts_drops_rows_without_abstract_or_label(tmp_path):
    path = make_csv(
        tmp_path,
        [
            {"doi": "a", "abstract": "has label", "SDG03": 1},
            {"doi": "b", "abstract": None, "SDG03": 1},
            {"doi": "c", "abstract": "no label"},
        ],
    )
    df = load_abstracts(path)
    assert df["abstract"].tolist() == ["has label"]
    assert list(df.columns) == ["abstract", *SDG_COLUMNS]
    assert df.loc[0, "SDG03"] == 1 and df[SDG_COLUMNS].sum(axis=1).item() == 1


def test_one_label_per_row():
    row = {c: 0 for c in SDG_COLUMNS} | {"abstract": "x", "SDG01": 1, "SDG13": 1}
    out = one_label_per_row(pd.DataFrame([row]))
    assert len(out) == 2
    assert out[SDG_COLUMNS].sum(axis=1).tolist() == [1, 1]
    assert set(out.loc[out["SDG13"] == 1, "abstract"]) == {"x"}


def test_split_is_disjoint_and_reproducible():
    rows = [{c: int(i % 17 == j) for j, c in enumerate(SDG_COLUMNS)} | {"abstract": f"t{i}"} for i in range(170)]
    df = pd.DataFrame(rows)[["abstract", *SDG_COLUMNS]]
    first, second = split(df, seed=1), split(df, seed=1)
    sizes = {k: len(v) for k, v in first.items()}
    assert sum(sizes.values()) == 170 and sizes["train"] > sizes["test"] > 0
    texts = [set(first[k]["abstract"]) for k in first]
    assert not (texts[0] & texts[1] or texts[0] & texts[2] or texts[1] & texts[2])
    assert first["test"]["abstract"] == second["test"]["abstract"]
