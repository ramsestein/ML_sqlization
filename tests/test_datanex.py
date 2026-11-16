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
    assert sql.count("IN (SELECT patient_ref FROM cohorte)") == 4  # episodes, rc, labs, demographics
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


def test_preparar(tmp_path):
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression
    import joblib
    import pandas as pd
    nombres = ["FC_mean", "age"]
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(80, 2)), columns=nombres)
    y = (X.FC_mean > 0).astype(int)
    joblib.dump(LogisticRegression().fit(X, y), tmp_path / "lr.joblib")
    p = E.Parametros(arboles=10, profundidad=2, bins=16)
    plan = DN.preparar([("lr", tmp_path / "lr.joblib")], tmp_path / "dn", p=p, arboles_por_consulta=4,
                       variables_entrenamiento=nombres)
    c = tmp_path / "dn" / "consultas"
    assert sorted(x.name for x in c.iterdir()) == ["00_variables_10_pacientes.txt", "01_inferencia_lr.txt", "entrenamiento_01.txt"]
    assert plan["entrenamiento_tandas"] == 3 and plan["variables"] == nombres
    for f in c.iterdir():
        t = f.read_text()
        assert len(t.encode()) < config.MAX_SQL_BYTES and "cohorte AS (" in t and t.startswith("-- ")
    assert "json_format(CAST(r AS JSON)) AS resultado_json" in (c / "entrenamiento_01.txt").read_text()
    assert (tmp_path / "dn" / "LEEME.md").exists() and (tmp_path / "dn" / "documentos" / "lr.json").exists()
