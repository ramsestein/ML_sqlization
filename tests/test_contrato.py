import pytest

from fnxsql.contrato import Contrato, interpretar, sql_variables, variables_del_modelo


@pytest.mark.parametrize("nombre, fuente, codigo, valor, est", [
    ("FC_mean", "rc", "FC", None, "mean"),
    ("TAS_last", "rc", "TAS", None, "last"),
    ("LAB2467_max", "labs", "LAB2467", None, "max"),
    ("ESTADO_CONCIENC_ESTADO_CONCIENC_1_last", "rc_codigo", "ESTADO_CONCIENC", "ESTADO_CONCIENC_1", "last"),
    ("CAR_RITM_CAR_RITM_13_std", "rc_codigo", "CAR_RITM", "CAR_RITM_13", "std"),
    ("age_min", "edad", None, None, "min"),
])
def test_gramatica(nombre, fuente, codigo, valor, est):
    v = interpretar(nombre)
    assert (v.fuente, v.codigo, v.valor, v.estadistico) == (fuente, codigo, valor, est)


def test_edad_sexo_y_constantes():
    assert interpretar("age").fuente == "edad"
    assert interpretar("sex").fuente == "sexo"
    assert interpretar("sex_2").valor == "2"
    c = Contrato(constantes={"index_last": 0})
    assert interpretar("index_last", c).constante == 0.0


@pytest.mark.parametrize("nombre", ["Creatinine", "FC_media", "P_ART_S_avg"])
def test_rechaza_nombres_fuera_de_gramatica(nombre):
    with pytest.raises(ValueError):
        interpretar(nombre)


def test_sql_variables_estructura():
    vs, err = variables_del_modelo(["FC_mean", "LAB2467_last", "ESTADO_CONCIENC_ESTADO_CONCIENC_2_max", "age", "sex_2",
                                    "FC_count"])
    assert not err
    sql = sql_variables(vs, Contrato(si_falta=0), [1, 2])
    assert "patient_ref IN (1, 2)" in sql and "stddev_samp" not in sql
    assert "max_by(l.result_num, l.extrac_date) FILTER (WHERE l.lab_sap_ref = 'LAB2467')" in sql
    assert "-- FC_mean\n" in sql  # el comentario va en su propia línea (no se come la coma)
    for i in range(len(vs)):
        assert f"AS v{i:03d}" in sql
