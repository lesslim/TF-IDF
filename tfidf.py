from typing import List, Set
from collections import Counter
from math import log

# Задание #1:


class CountVectorizer:
    """
    Класс для создания массива токенов и матрицы частоты
    вхождения слов из корпуса в этот массив.
    """

    def __init__(self, lowercase: bool = True):
        self.lowercase = lowercase
        self.features_set: Set[str] = set()
        self.feature_names: List[str] = []
        self.matrix: List[List[int]] = []

    def _fit(self, corpus: List[str], anew: bool):
        """
        Создание массива токенов, подготовка корпуса.
        """
        if anew:
            self.features_set = set()
            self.feature_names = []
        list_of_counters = []
        self.matrix = []

        for text in corpus:
            if self.lowercase:
                text = text.lower()
            words = text.split()
            list_of_counters.append(Counter(words))

            for word in words:
                if word in self.features_set:
                    continue
                self.features_set.add(word)
                self.feature_names.append(word)

        return list_of_counters

    def fit_transform(self, corpus: List[str], anew: bool = True):
        """
        Возвращает матрицу частоты вхождений слов в тексты корпуса.
        """
        list_of_counters = self._fit(corpus, anew)
        features_len = len(self.feature_names)
        for cntr in list_of_counters:
            ans = [0] * features_len

            for i in cntr:
                ans[self.feature_names.index(i)] = cntr[i]
            self.matrix.append(ans)
        return self.matrix

    def get_feature_names(self):
        """
        Возвращает массив токенов.
        """
        return self.feature_names


# Задание #2 term frequency
def tf_transform(matrix: List[List[int]]) -> List[List[float]]:
    """
    По матрице частоты вхождений слов в корпус получаем
    матрицу отношений числа вхождений к общему количеству слов.
    """
    return [[count / sum(row) for count in row] for row in matrix]


# Задание #3: inverse document-frequency
def idf_transform(matrix: List[List[int]]) -> List[float]:
    """
    Из матрицы встречаемости слов делаем матрицу инверсий
    частот встречаемости слов в документах корпуса.
    """
    word_exist = [[count > 0 for count in row] for row in matrix]
    docs_with_word = [sum(i) for i in zip(*word_exist)]
    all_docs = len(matrix)
    return [round(log((all_docs + 1) / (i + 1)) + 1, 3) for i in docs_with_word]


# Задание #4: tf-idf transformer
class TfidfTransformer:
    """
    По матрице частоты вхождений слов в корпус вычисляет tf-idf.
    """

    @staticmethod
    def tf_transform(matrix: List[List[int]]) -> List[List[float]]:
        """
        По матрице частоты вхождений слов в корпус получаем
        матрицу отношений числа вхождений к общему количеству слов.
        """
        return [[round(count / sum(row), 3) for count in row] for row in matrix]

    @staticmethod
    def idf_transform(matrix: List[List[int]]) -> List[float]:
        """
        Из матрицы встречаемости слов делаем матрицу инверсий
        частот встречаемости слов в документах корпуса.
        """
        word_exist = [[count > 0 for count in row] for row in matrix]
        docs_with_word = [sum(i) for i in zip(*word_exist)]
        all_docs = len(matrix)
        return [round(log((all_docs + 1) / (i + 1)) + 1, 3) for i in docs_with_word]

    @staticmethod
    def fit_transform(matrix: List[List[int]]) -> List[List[float]]:
        """
        По матрице частоты вхождений слов в корпус вычисляет tf-idf.
        """
        tf_matrix = tf_transform(matrix)
        idf_matrix = idf_transform(matrix)
        return [
            [round(tf * idf, 3) for (tf, idf) in zip(row, idf_matrix)]
            for row in tf_matrix
        ]


# Задание #5: tf-idf vectorizer
class TfidfVectorizer(CountVectorizer):
    """
    По корпусу вычисляем матрицу tf-idf.
    """

    def fit_transform(self, corpus: List[str]) -> List[List[float]]:
        """
        По корпусу вычисляем матрицу tf-idf.
        """
        matrix = super().fit_transform(corpus)
        return TfidfTransformer().fit_transform(matrix)
