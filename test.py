from tfidf import CountVectorizer, TfidfTransformer, TfidfVectorizer


def test_CountVectorizer():
    corpus = [
        "Crock Pot Pasta Never boil pasta again",
        "Pasta Pomodoro Fresh ingredients Parmesan to taste",
    ]
    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(corpus)
    assert vectorizer.get_feature_names() == [
        "crock",
        "pot",
        "pasta",
        "never",
        "boil",
        "again",
        "pomodoro",
        "fresh",
        "ingredients",
        "parmesan",
        "to",
        "taste",
    ]
    assert count_matrix == [
        [1, 1, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    ]
    corpus2 = ["Crock not to taste"]
    assert vectorizer.fit_transform(corpus2, False) == [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
    ]
    assert vectorizer.get_feature_names() == [
        "crock",
        "pot",
        "pasta",
        "never",
        "boil",
        "again",
        "pomodoro",
        "fresh",
        "ingredients",
        "parmesan",
        "to",
        "taste",
        "not",
    ]


def test_TfidfTransformer():
    count_matrix = [
        [1, 1, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    ]
    transformer = TfidfTransformer()
    assert transformer.fit_transform(count_matrix) == [
        [0.201, 0.201, 0.286, 0.201, 0.201, 0.201, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.143, 0.0, 0.0, 0.0, 0.201, 0.201, 0.201, 0.201, 0.201, 0.201],
    ]


def test_TfidfVectorizer():
    corpus = [
        "Crock Pot Pasta Never boil pasta again",
        "Pasta Pomodoro Fresh ingredients Parmesan to taste",
    ]
    v1 = TfidfVectorizer()
    assert v1.fit_transform(corpus) == [
        [0.201, 0.201, 0.286, 0.201, 0.201, 0.201, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.143, 0.0, 0.0, 0.0, 0.201, 0.201, 0.201, 0.201, 0.201, 0.201],
    ]


if __name__ == "__main__":
    test_CountVectorizer()
    test_TfidfTransformer()
    test_TfidfVectorizer()
