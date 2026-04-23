import pandas as pd
from etl import transform


def test_normal():
    df = pd.DataFrame({
        "name.first": ["John"],
        "name.last": ["Doe"],
        "dob.age": [25],
        "dob.date": ["2000-01-01"],
        "email": ["test@gmail.com"]
    })

    res = transform(df)
    assert res.iloc[0]["age_group"] == "Young Adult"
    assert res.iloc[0]["email_domain"] == "gmail.com"


def test_duplicates():
    df = pd.DataFrame({
        "name.first": ["A","A"],
        "name.last": ["B","B"],
        "dob.age": [20,20],
        "dob.date": ["2000","2000"],
        "email": ["a@mail.com","a@mail.com"]
    })

    res = transform(df)
    assert len(res) == 1


def test_missing_email():
    df = pd.DataFrame({
        "name.first": ["A"],
        "name.last": ["B"],
        "dob.age": [20],
        "dob.date": ["2000"],
        "email": [None]
    })

    res = transform(df)
    assert len(res) == 0


def test_empty():
    df = pd.DataFrame()
    res = transform(df)
    assert res.empty