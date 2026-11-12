"""Precisión y binario (binario.py), sin servidor: redondeos, cuantización y una emulación en Python de lo que hace la
consulta (descodificar los literales base64 y recorrer los árboles / multiplicar las capas con las mismas fórmulas),
comparada con el documento cuantizado (debe ser exacta) y con el modelo original (el error de la precisión)."""
import base64
import math
import re

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fnxsql import binario as B, ir, sql
from test_ir import datos, doc_de

TIPOS = {1: ">u1", 2: ">u2", 4: ">u4"}


def test_redondeos_hacia_fuera():
    for t in [74.1, 1.38, 0.04, 1.27, -3.3, 1e-7, 123456.789]:
        up, dn = B.f32_arriba(t), B.f32_abajo(t)
        assert float(np.float32(up)) == up and float(np.float32(dn)) == dn
        # son los float32 contiguos a t (comparación en double: NumPy 2 bajaría t a float32 frente a un np.float32)
        assert dn <= t <= up
        assert float(np.nextafter(np.float32(up), np.float32(-np.inf))) < t < float(np.nextafter(np.float32(dn), np.float32(np.inf)))
        assert (t <= B.umbral(t, "<=")) and not (t < B.umbral(t, "<"))  # un valor igual al umbral queda del mismo lado
    assert B.f32_arriba(math.inf) == math.inf and B.umbral(1.7976931348623157e308, "<=") == math.inf


def test_cuantizar_int8():
    assert B.escala_int8([0.5, -1.27]) == B.f32(1.27 / 127) and B.escala_int8([0, 0]) == 1.0
    s = B.escala_int8([1.0])
    assert B.q_int8(1.0, s) == 127 and B.q_int8(-5.0, s) == -127 and B.q_int8(0.004, s) == 1
    with pytest.raises(ValueError):
        B.cuantizar({"nucleo": {}}, "int4")


def _blob(s, alias):
    return base64.b64decode(re.search(r"from_base64\('([^']*)'\) AS " + alias + r"\b", s).group(1))


def _ancho(s, col):
    pad = re.search(r"from_big_endian_32\((?:X'(0*)' \|\| )?substr\(" + col + r"\b", s).group(1) or ""
    return 4 - len(pad) // 2


def _ints(s, col):
    return np.frombuffer(_blob(s, col), dtype=TIPOS[_ancho(s, col)]).astype(np.int64)


def _emular_arboles(s, dq, filas):
    """Recorre los árboles como la consulta: f, mg, d del código entero; umbrales y hojas del binario."""
    c, raices = _ints(s, "bc"), _ints(s, "br")
    m = int(re.search(r"transform\(c, z -> z % (\d+)\) AS d", s).group(1))
    prof = int(re.search(r"reduce\(sequence\(1, (\d+)\), element_at\(a\.raices, t\)", s).group(1))
    f, mg, d = c // (2 * m), (c // m) % 2, c % m
    if " AS bu" in s:  # int8: índice en la tabla de umbrales (internos) o q + 128 (hojas)
        u = _ints(s, "bu")
        ut = np.frombuffer(_blob(s, "bt"), dtype=">f4").astype(np.float64)
        esc = np.frombuffer(_blob(s, "be"), dtype=">f4").astype(np.float64)
        umb = np.where(f > 0, ut[np.minimum(u, len(ut) - 1)], 0.0)
        hoja = lambda n, t: (u[n - 1] - 128) * esc[t]
    else:
        umb = np.frombuffer(_blob(s, "bv"), dtype=">f4" if "from_ieee754_32(substr(bv" in s else ">f8").astype(np.float64)
        hoja = lambda n, t: umb[n - 1]
    op = "<" if " < element_at(a.v, n)" in s else "<="
    nuc = dq["nucleo"]
    ef32 = bool(nuc.get("entrada_float32"))  # x como real (árboles de scikit-learn), con cualquier precisión
    salida = []
    for fila in filas:
        x = [None if (fila[n] is None or (isinstance(fila[n], float) and math.isnan(fila[n]))) else fila[n]
             for n in dq["entrada_nucleo"]]
        hojas = []
        for t, rz in enumerate(raices):
            n = int(rz)
            for _ in range(prof):
                if f[n - 1] == 0:
                    break
                xv = x[f[n - 1] - 1]
                xv = float(np.float32(xv)) if (ef32 and xv is not None) else xv
                izq = (mg[n - 1] == 1) if xv is None else ((xv < umb[n - 1]) if op == "<" else (xv <= umb[n - 1]))
                n = n + d[n - 1] + (0 if izq else 1)
            assert f[n - 1] == 0, "el recorrido no llega a una hoja"
            hojas.append(hoja(n, t))
        salida.append(sum(hojas) / len(hojas) if nuc["combinacion"] == "media" else ir.sigmoide(nuc["base"] + sum(hojas)))
    return np.array(salida)


TOL_ORIGINAL = {"float64": 1e-12, "float32": 1e-6, "int8": 0.05}


@pytest.mark.parametrize("modelo", [lambda: RandomForestClassifier(n_estimators=8, max_leaf_nodes=24, random_state=0),
                                    lambda: HistGradientBoostingClassifier(max_iter=30, max_leaf_nodes=12, random_state=0)])
@pytest.mark.parametrize("precision", ["float64", "float32", "int8"])
def test_arboles_binarios_emulados(modelo, precision):
    X, y = datos()
    m = modelo().fit(X, y)
    doc = doc_de(m)
    dq = B.cuantizar(doc, precision)
    s, info = sql.compilar(doc, [1], "arrays", precision=precision)
    assert "json_parse" not in s and (("from_ieee754_64" in s) == (precision == "float64"))
    filas = X.to_dict("records")
    p_sql = _emular_arboles(s, dq, filas)
    p_dq = np.array([ir.evaluar(dq, r) for r in filas])
    p_mod = m.predict_proba(X)[:, 1]
    assert np.abs(p_sql - p_dq).max() <= 1e-12  # el SQL calcula exactamente el documento cuantizado
    assert np.abs(p_dq - p_mod).max() <= TOL_ORIGINAL[precision]  # error de la precisión frente al original
    assert np.abs(np.array([ir.evaluar(doc, r) for r in filas]) - p_mod).max() <= 1e-12  # if-then exacto


@pytest.mark.parametrize("precision", ["float64", "float32", "int8"])
def test_red_binaria_emulada(precision):
    X, y = datos(ausentes=False)
    m = Pipeline([("s", StandardScaler()), ("m", MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=300, random_state=0))]).fit(X, y)
    doc = doc_de(m)
    dq = B.cuantizar(doc, precision)
    s, _ = sql.compilar(doc, [1], "arrays", precision=precision)
    filas = X.to_dict("records")
    for li, capa in enumerate(dq["nucleo"]["capas"], start=1):
        nf, nc = len(capa["pesos"]), len(capa["pesos"][0])
        if precision == "int8":
            q = np.frombuffer(_blob(s, f"bw{li}"), dtype=">u1").astype(np.int64).reshape(nf, nc) - 128
            esc = np.frombuffer(_blob(s, f"bs{li}"), dtype=">f4").astype(np.float64)
            W = q * esc[:, None]
            assert np.abs(q).max() <= 127
        else:
            W = np.frombuffer(_blob(s, f"bw{li}"), dtype=">f4" if precision == "float32" else ">f8").astype(np.float64).reshape(nf, nc)
        assert np.array_equal(W, np.array(capa["pesos"]))  # el binario guarda exactamente los pesos cuantizados
    p_dq = np.array([ir.evaluar(dq, r) for r in filas])
    assert np.abs(p_dq - m.predict_proba(X)[:, 1]).max() <= TOL_ORIGINAL[precision]


def test_float32_e_int8_ocupan_menos():
    X, y = datos()
    doc = doc_de(RandomForestClassifier(n_estimators=30, max_leaf_nodes=32, random_state=0).fit(X, y))
    b = {p: sql.compilar(doc, [1], "arrays", precision=p)[1]["bytes"] for p in ("float64", "float32", "int8")}
    assert b["int8"] < b["float32"] < b["float64"]
