"""Compilación SQL sin servidor: tipos admitidos, salvaguardas y contenido básico."""
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fnxsql import config, sql
from test_ir import datos, doc_de


def test_tipos_y_contenido():
    X, y = datos()
    rf = doc_de(RandomForestClassifier(n_estimators=5, max_leaf_nodes=8, random_state=0).fit(X, y))
    s, info = sql.compilar(rf, [1, 2], "arrays")
    assert "reduce(sequence(1," in s and "json_parse" in s and info["arboles"] == 5
    s2, info2 = sql.compilar(rf, [1], "case")
    assert "ARRAY[a0000" in s2  # segunda referencia a cada árbol (evita la fusión de expresiones en Trino)
    lr = doc_de(Pipeline([("i", SimpleImputer()), ("s", StandardScaler()), ("lr", LogisticRegression())]).fit(X, y))
    s3, _ = sql.compilar(lr, [1], "auto")
    assert "coalesce(v000," in s3 and "exp(-(" in s3
    with pytest.raises(ValueError):
        sql.compilar(lr, [1], "arrays")


def test_limite_de_terminos(monkeypatch):
    X, y = datos(ausentes=False)
    red = doc_de(Pipeline([("s", StandardScaler()), ("m", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=50))]).fit(X, y))
    monkeypatch.setattr(config, "MAX_TERMINOS_SQL", 100)
    with pytest.raises(sql.SQLDemasiadoGrande):
        sql.compilar(red, [1], "terminos")
    s, info = sql.compilar(red, [1], "arrays")  # con arrays no hay términos sueltos
    assert info["pesos"] > 100 and "array(array(double))" in s
