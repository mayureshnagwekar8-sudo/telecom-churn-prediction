import pandas as pd

from churn import add_tenure_bin, clean, encode


def test_clean_drops_blank_total_charges():
    df = pd.DataFrame({"TotalCharges": ["29.85", " ", "100.5"]})
    out = clean(df)
    assert len(out) == 2
    assert out["TotalCharges"].dtype == float


def test_tenure_bins_edges():
    df = add_tenure_bin(pd.DataFrame({"tenure": [0, 1, 12, 13, 72]}))
    assert df["tenure_bin"].tolist() == ["1-12", "1-12", "1-12", "13-24", "61-72"]


def test_encode_aligns_to_training_columns():
    columns = ["MonthlyCharges", "Contract_One year", "Contract_Two year", "tenure_bin_13-24"]
    row = pd.DataFrame([{"tenure": 20, "MonthlyCharges": 70.0, "Contract": "Two year"}])
    X = encode(row, columns=columns)
    assert list(X.columns) == columns
    # Regression test: drop_first on a single row used to zero out every category
    assert X.loc[0, "Contract_Two year"] == 1
    assert X.loc[0, "Contract_One year"] == 0
    assert X.loc[0, "tenure_bin_13-24"] == 1


def test_single_row_matches_training_encoding():
    df = clean(pd.read_csv("Customer-Churn.csv"))
    X_train = encode(df)
    X_row = encode(df.iloc[[0]], columns=list(X_train.columns))
    pd.testing.assert_series_equal(
        X_row.iloc[0], X_train.iloc[0], check_names=False
    )


def test_torch_model_shapes_and_loss():
    import torch
    from torch import nn
    from torch_train import ChurnMLP

    model = ChurnMLP(n_features=34)
    x = torch.randn(64, 34)
    logits = model(x)
    assert logits.shape == (64, 1)
    loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.76]))(logits, torch.ones(64, 1))
    assert torch.isfinite(loss)
