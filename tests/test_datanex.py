"""DataNex sin servidor: cohorte por hash, consultas generadas y comprobación de los CSV (también formateados)."""
import csv
import zipfile

import numpy as np
import pandas as pd
import pytest

from fnxsql import config, contrato as K, datanex as DN, ir


def test_hash_sin_desbordar():
    # (patient_ref * 2654435761 + d) mod 2^32 para patient_ref < 2^31, y sin desbordar bigint con cualquier otro
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


def test_leer_csv_de_athena(tmp_path):
    ruta = tmp_path / "x.csv"
    with open(ruta, "w", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["PATIENT_REF", "probabilidad", "v000"])
        w.writerow(["7", "0.25", ""])
    assert DN.leer_csv(ruta) == [{"patient_ref": "7", "probabilidad": "0.25", "v000": None}]


def test_numeros_formateados():
    assert DN._num("1,521,782") == 1521782 and DN._num("95.06") == 95.06 and DN._num(None) is None
    assert DN._id("1,521,782") == "1521782" and DN._id("1521782") == "1521782"
    assert DN._decimales("0.0085") == 4 and DN._decimales("0.02") == 2 and DN._decimales("1") == 0
    assert DN._decimales("6.7e-05") == 6 and DN._decimales("0.012345678901234568") is None


def _preparar(tmp_path):
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression
    import joblib
    nombres = ["FC_mean", "age"]
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(80, 2)) * [10, 15] + [80, 60], columns=nombres)
    y = (X.FC_mean > 80).astype(int)
    joblib.dump(LogisticRegression().fit(X, y), tmp_path / "lr.joblib")
    return DN.preparar([("lr", tmp_path / "lr.joblib")], tmp_path / "dn")


def test_preparar(tmp_path):
    plan = _preparar(tmp_path)
    c = tmp_path / "dn" / "consultas"
    assert sorted(x.name for x in c.iterdir()) == ["00_variables_10_pacientes.txt", "01_inferencia_lr.txt", "LEEME.txt"]
    for f in c.glob("0*.txt"):
        t = f.read_text()
        # autocontenidas: eligen sus pacientes dentro (cohorte) y caben en Athena
        assert len(t.encode()) < config.MAX_SQL_BYTES and "cohorte AS (" in t and t.startswith("-- ")
    assert plan["variables"] == ["FC_mean", "age"] and "entrenamientos" not in plan
    assert len(zipfile.ZipFile(tmp_path / "dn" / "consultas_datanex.zip").namelist()) == 3


def test_comprobar_csv_exacto_y_formateado(tmp_path):
    _preparar(tmp_path)
    import json
    doc = json.loads((tmp_path / "dn" / "documentos" / "lr.json").read_text())  # lo que calcula el SQL (float32)
    vals = [(101, 1001, 85.123456789, 70.4), (101, 1002, None, 70.5), (2203, 1_521_782, 60.0, 41.0)]
    p = [ir.evaluar(doc, {"FC_mean": a, "age": b}) for _, _, a, b in vals]
    var = pd.DataFrame([{"patient_ref": str(pr), "episode_ref": str(ep), "t0": "", "t_eval": "", "v000": "" if a is None else repr(a),
                         "v001": repr(b)} for pr, ep, a, b in vals])
    exacto, formateado = tmp_path / "exacto", tmp_path / "formateado"
    for d, fmt_id, fmt_p in ((exacto, str, repr), (formateado, lambda x: f"{x:,}", lambda x: f"{x:.2g}")):
        d.mkdir()
        var.assign(episode_ref=[fmt_id(ep) for _, ep, _, _ in vals]).to_csv(d / "00_variables_10_pacientes.csv", index=False)
        pd.DataFrame({"patient_ref": [str(v[0]) for v in vals], "episode_ref": [fmt_id(v[1]) for v in vals], "ingreso": "",
                      "momento": "", "probabilidad": [fmt_p(x) for x in p]}).to_csv(d / "01_inferencia_lr.csv", index=False)
        out = DN.comprobar(tmp_path / "dn", d)
        assert (out.resultado == "PASA").all(), out.to_string()
    # una probabilidad equivocada se detecta también en el CSV redondeado
    pd.DataFrame({"patient_ref": [str(v[0]) for v in vals], "episode_ref": [f"{v[1]:,}" for v in vals], "ingreso": "",
                  "momento": "", "probabilidad": [f"{x * 1.5:.2g}" for x in p]}).to_csv(formateado / "01_inferencia_lr.csv", index=False)
    assert (DN.comprobar(tmp_path / "dn", formateado).resultado == "FALLA").any()
