"""Documento if-then de una métrica PERISIM exportada (perisim: src/deploy/fnxsql_export.py, formato perisim_fnxsql_v1).

El artefacto trae, por dimensión de la métrica, un paso tabla / interpolar / categoria de una entrada (la función de
forma ya dividida por su escala), el peso NCA de cada dimensión, k, el alfa del voto suavizado y las reglas de la bolsa.
Aquí solo se cambian los nombres de PERISIM por variables del contrato y se monta el núcleo "vecinos" (suavizado,
euclídeo con pesos): la bolsa es la propia base del hospital, como en cualquier KNN de fnxsql.
"""
from . import ir
from .contrato import variables_del_modelo

FORMATO = "perisim_fnxsql_v1"


def documento(artefacto, nombres, contrato, nombre_modelo="perisim"):
    """nombres: entrada de PERISIM -> variable del contrato (gramática de contrato.py)."""
    _formato(artefacto)
    falta = [e for e in artefacto["entradas"] if e not in nombres]
    if falta:
        raise ValueError(f"entradas de PERISIM sin variable del contrato: {falta}")
    art = dict(artefacto, entradas=[nombres[e] for e in artefacto["entradas"]],
               preproceso=[dict(p, variable=nombres[p["variable"]]) for p in artefacto["preproceso"]])
    vs, err = variables_del_modelo(art["entradas"], contrato)
    if err:
        raise ValueError(f"variables no válidas para el contrato: {err}")
    return documento_de_variables(art, vs, contrato, nombre_modelo)


def _formato(artefacto):
    if artefacto.get("formato") != FORMATO:
        raise ValueError(f"artefacto con formato {artefacto.get('formato')!r} (se espera {FORMATO!r})")


def documento_de_variables(artefacto, variables, contrato, nombre_modelo="perisim"):
    """Documento con las variables del contrato ya resueltas (las entradas del artefacto ya tienen sus nombres)."""
    _formato(artefacto)
    if contrato.si_falta is not None:
        raise ValueError("contrato con si_falta numérico: PERISIM trata los ausentes en sus propios pasos (si_falta = null)")
    if [v.nombre for v in variables] != list(artefacto["entradas"]):
        raise ValueError("las variables no coinciden con las entradas del artefacto")
    for p in artefacto["preproceso"]:
        if p["tipo"] not in ir.FORMAS:
            raise ValueError(f"paso {p['tipo']!r} no esperado en un artefacto PERISIM")
    bolsa = ir.bolsa_de(contrato)
    bolsa.update(un_vecino_por_paciente=bool(artefacto["un_vecino_por_paciente"]),
                 excluir_mismo_paciente=bool(artefacto.get("excluir_mismo_paciente", True)))
    return {"formato": ir.FORMATO, "modelo": {"nombre": nombre_modelo, "origen": "perisim", "clase": "PERISIM",
                                              "valores_de_entrenamiento": bool(artefacto.get("valores_de_entrenamiento"))},
            "contrato": contrato.a_dict(),
            "variables": [{k: getattr(v, k) for k in ("nombre", "fuente", "codigo", "valor", "estadistico", "constante")
                           if getattr(v, k) is not None} for v in variables],
            "preproceso": [dict(p) for p in artefacto["preproceso"]],
            "entrada_nucleo": [p["salida"] for p in artefacto["preproceso"]],
            "nucleo": {"tipo": "vecinos", "k": int(artefacto["k"]), "ponderacion": "suavizada", "p": 2,
                       "pesos": [float(w) for w in artefacto["pesos"]], "alfa": float(artefacto["alfa"]),
                       "prior": None, "bolsa": bolsa}}
