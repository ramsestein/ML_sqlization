"""DataNex sin servidor: cohorte por hash, lectura de resultados (JSON en CSV de Athena) y consultas generadas."""
import csv
import json

import numpy as np
import pytest

from fnxsql import config, contrato as K, datanex as DN, entrenar as E, entrenar_una as U


def test_hash_sin_desbordar():
    # el de siempre para patient_ref < 2^31 (FakeNex) y sin desbordar bigint para cualquier otro
    assert K.hash_py([1_000_000])[0] == (1_000_000 * 2654435761) % 2 ** 32
    grande = 98_765_432_101
    assert K.hash_py([grande])[0] == ((grande % 2 ** 31) * 2654435761) % 2 ** 32
    assert (2 ** 31 - 1) * 2654435761 + 2 ** 32 < 2 ** 63  # el producto de la consulta cabe en bigint
    assert "mod(mod(patient_ref, 2147483648) * 2654435761 + " in K.hash_sql("patient_ref", 1)


def test_cohorte_en_las_consultas():
    vs, _ = K.variables_del_modelo(["FC_mean", "LAB2467_max", "age"])
    co = K.Cohorte(10, 1)
    sql = K.sql_variables(vs, K.Contrato(), None, co)
    assert sql.startswith("WITH\ncohorte AS (") and "LIMIT 10" in sql
    assert "< 1000000000" not in sql  # en DataNex no se filtra la base original de FakeNex
    assert sql.count("IN (SELECT patient_ref FROM cohorte)") == 1  # solo en episodes: los JOIN con adm ya restringen
    assert "IN (SELECT patient_ref FROM cohorte)" in K.sql_etiqueta(None, co)


def test_leer_fila_objetos_o_arrays():
    arb = {"feat": [1, 0, 0], "bin": [3, 0, 0], "izq": [True, False, False], "val": [None, 0.5, -0.25],
           "g": [1.0, 0.5, 0.5], "h": [2.0, 1.0, 1.0], "gan": [3.0, 0.0, 0.0], "n": [10, 6, 4]}
    perd = [{"e": 0.3, "p": 0.31}, {"e": 0.2, "p": 0.25}]
    base = {"n": "10", "n_prueba": "3", "positivos": "2.0", "m0": "-1.5", "cortes_json": "[[1.0, 2.0]]", "kv_json": "[2]"}
    como_objeto = dict(base, resultado_json=json.dumps({"arb": [arb], "perd": perd}))
    como_array = dict(base, resultado_json=json.dumps([[list(arb.values())], [list(q.values()) for q in perd]]))
    a, b = U.leer_fila(como_objeto), U.leer_fila(como_array)
    assert a == b and a[0] == 10 and a[6][0][0] == [1, 0, 0] and a[7] == [(0.3, 0.31), (0.2, 0.25)]


def test_leer_csv_de_athena(tmp_path):
    ruta = tmp_path / "x.csv"
    with open(ruta, "w", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["PATIENT_REF", "probabilidad", "v000"])
        w.writerow(["7", "0.25", ""])
    assert DN.leer_csv(ruta) == [{"patient_ref": "7", "probabilidad": "0.25", "v000": None}]


def test_preparar_e_importar(tmp_path):
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression
    import joblib
    import pandas as pd
    nombres = ["FC_mean", "age"]
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(80, 2)), columns=nombres)
    y = (X.FC_mean > 0).astype(int)
    joblib.dump(LogisticRegression().fit(X, y), tmp_path / "lr.joblib")
    plan = DN.preparar([("lr", tmp_path / "lr.joblib")], tmp_path / "dn", entrenamientos=((4, 2),), variables_entrenamiento=nombres)
    c = tmp_path / "dn" / "consultas"
    assert sorted(x.name for x in c.iterdir()) == ["00_variables_10_pacientes.txt", "01_inferencia_lr.txt",
                                                   "02_entrenamiento_4_arboles.txt", "LEEME.txt"]
    for f in c.glob("*_*.txt"):
        t = f.read_text()
        # autocontenidas: eligen sus pacientes dentro (cohorte) y caben en Athena
        assert len(t.encode()) < config.MAX_SQL_BYTES and "cohorte AS (" in t and t.startswith("-- ")
    e = (c / "02_entrenamiento_4_arboles.txt").read_text()
    assert "CROSS JOIN UNNEST(m.r.arb) WITH ORDINALITY" in e and "json_format" not in e
    import zipfile
    assert len(zipfile.ZipFile(tmp_path / "dn" / "consultas_datanex.zip").namelist()) == 4
    # importar: una tabla de nodos como la que devuelve la consulta -> modelo exacto
    p = E.Parametros(arboles=2, profundidad=2)
    filas = []
    for t, (hoja_i, hoja_d) in enumerate(((0.25, -0.125), (0.5, -0.0625)), start=1):
        com = {"perdida_entrenamiento": 0.3 - 0.1 * t, "perdida_prueba": 0.35 - 0.1 * t, "margen_base": -1.5,
               "ingresos_entrenamiento": 80, "ingresos_prueba": 20, "muertes_entrenamiento": 30,
               "perdida_base_entrenamiento": 0.4, "perdida_base_prueba": 0.45, "ingresos": 10, "hessiano": 2.0}
        filas += [dict(com, arbol=t, nodo=0, variable="FC_mean", umbral=0.5, ausentes_izquierda="true", ganancia=1.0),
                  dict(com, arbol=t, nodo=1, hoja=hoja_i), dict(com, arbol=t, nodo=2, hoja=hoja_d)]
    ruta = tmp_path / "02_entrenamiento_4_arboles.csv"
    pd.DataFrame(filas).to_csv(ruta, index=False)
    res, hist = DN.importar(ruta, tmp_path / "dn")
    doc = res["documento"]
    assert len(doc["nucleo"]["arboles"]) == 2 and doc["nucleo"]["base"] == -1.5 and hist[-1]["perdida_prueba"] == pytest.approx(0.15)
    from fnxsql import ir
    assert ir.evaluar(doc, {"FC_mean": 0.0, "age": 50}) == pytest.approx(1 / (1 + np.exp(-(-1.5 + 0.25 + 0.5))), abs=1e-6)
    assert ir.evaluar(doc, {"FC_mean": None, "age": 50}) == ir.evaluar(doc, {"FC_mean": 0.0, "age": 50})  # ausentes a la izquierda
