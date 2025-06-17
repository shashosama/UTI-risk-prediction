def test_symptom_score():
    df = pd.DataFrame({
        "Occurrence of nausea": [True, False],
        "Lumbar pain": [False, True],
        "Urine pushing (continuous need for urination)": [True, True],
        "Micturition pains": [False, False],
        "Burning of urethra, itch, swelling of urethra outlet": [False, True]
    })
    df = preprocess(df)
    assert all(df["symptom_score"] >= 0)
