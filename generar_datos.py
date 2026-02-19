import pandas as pd
import numpy as np
import random
from datetime import timedelta

# Aseguramos resultados consistentes
np.random.seed(42)
random.seed(42)

# ==========================================
# 1. PARÁMETROS DEL PROCESO Y VARIANTES
# ==========================================
variants_distribution = [
    ("V1", 300, ["Ingreso de requerimiento", "Admisibilidad", "Tramitación tipo 1", "Revisión", "Firma"]),
    ("V2", 250, ["Ingreso de requerimiento", "Admisibilidad", "Tramitación tipo 2", "Revisión", "Firma"]),
    ("V3", 200, ["Ingreso de requerimiento", "Admisibilidad", "Tramitación tipo 3", "Revisión", "Firma"]),
    ("V4", 50,  ["Ingreso de requerimiento", "Admisibilidad", "No admisible"]),
    ("V5", 80,  ["Ingreso de requerimiento", "Admisibilidad", "Tramitación tipo 1", "Revisión", "Devolver a tramitación", "Tramitación tipo 1", "Revisión", "Firma"]),
    ("V6", 50,  ["Ingreso de requerimiento", "Admisibilidad", "Tramitación tipo 2", "Revisión", "Devolver a tramitación", "Tramitación tipo 2", "Revisión", "Firma"]),
    ("V7", 30,  ["Ingreso de requerimiento", "Admisibilidad", "Tramitación tipo 3", "Revisión", "Devolver a tramitación", "Tramitación tipo 3", "Revisión", "Firma"]),
    ("V8", 20,  ["Ingreso de requerimiento", "Admisibilidad", "Tramitación tipo 1", "Revisión", "Firma", "Devuelve a revisión", "Revisión", "Firma"]),
    ("V9", 10,  ["Ingreso de requerimiento", "Admisibilidad", "Tramitación tipo 2", "Revisión", "Firma", "Devuelve a revisión", "Revisión", "Firma"]),
    ("V10", 10, ["Ingreso de requerimiento", "Admisibilidad", "Tramitación tipo 1", "Revisión", "Devolver a tramitación", "Tramitación tipo 1", "Revisión", "Firma", "Devuelve a revisión", "Revisión", "Firma"])
]

state_resources = {
    "Ingreso de requerimiento": ["Tramitador A"],
    "Admisibilidad": ["Jefe Causas"],
    "No admisible": ["Jefe Causas"],
    "Tramitación tipo 1": ["Tramitador A"],
    "Tramitación tipo 2": ["Tramitador B", "Tramitador C"],
    "Tramitación tipo 3": ["Tramitador A", "Tramitador B", "Tramitador C"],
    "Revisión": ["Jefe Causas"],
    "Devolver a tramitación": ["Jefe Causas"],
    "Firma": ["Juez"],
    "Devuelve a revisión": ["Juez"]
}

# Probabilidad de exceder el SLA (Diferente productividad)
resource_sla_exceed_prob = {"Tramitador A": 0.15, "Tramitador B": 0.20, "Tramitador C": 0.25}

log_data = []
case_id_counter = 1
start_date_range = pd.date_range(start="2023-01-01", end="2023-10-01")

print("Generando 1000 casos y calculando SLAs... Esto tomará un par de segundos.")

# ==========================================
# 2. GENERACIÓN DEL LOG DE EVENTOS
# ==========================================
for variant_name, count, path in variants_distribution:
    for _ in range(count):
        case_id = f"CASO_{case_id_counter:04d}"
        case_id_counter += 1
        current_date = random.choice(start_date_range)
        
        # Asignar un recurso fijo para el caso si hay múltiples opciones
        case_resources = {}
        for state in set(path):
            if state in ["Tramitación tipo 1", "Tramitación tipo 2", "Tramitación tipo 3"]:
                case_resources[state] = random.choice(state_resources[state])
            else:
                case_resources[state] = state_resources[state][0]
                
        for idx, state in enumerate(path):
            res = case_resources[state]
            log_data.append({
                "ID": case_id,
                "TIPO RECURSO": "RECURSO_TIPO_1",
                "DESCRIPCION RECURSO": "Descripción de Prueba",
                "GLS_SALA_RECURSO": "Sala Primera",
                "FECHA_ESTADO": current_date.strftime("%d-%m-%Y"),
                "ESTADO": state,
                "RESPONSABLE": res
            })
            
            # Calcular fechas (Evitando multas de SLA en la medida de lo posible y saltando fines de semana)
            if idx < len(path) - 1:
                next_state = path[idx+1]
                delta_days = 0
                if state == "Tramitación tipo 1":
                    delta_days = random.randint(2, 4) if random.random() < resource_sla_exceed_prob.get(res, 0.2) else random.randint(0, 1)
                elif state == "Tramitación tipo 2":
                    delta_days = random.randint(3, 5) if random.random() < resource_sla_exceed_prob.get(res, 0.2) else random.randint(1, 2)
                elif state == "Tramitación tipo 3":
                    delta_days = random.randint(6, 10) if random.random() < resource_sla_exceed_prob.get(res, 0.2) else random.randint(2, 5)
                elif state in ["Admisibilidad", "Revisión", "Firma"]:
                    delta_days = random.randint(1, 3)
                else:
                    delta_days = random.randint(0, 1)
                
                current_date += timedelta(days=delta_days)
                while current_date.weekday() > 4: # Ignorar Sábado y Domingo
                    current_date += timedelta(days=1)

df_new_log = pd.DataFrame(log_data)
df_new_log.to_csv("nuevo_log_procesos.csv", sep=";", index=False, encoding='utf-8-sig')

# ==========================================
# 3. GENERACIÓN DEL MAESTRO DE ESTADOS
# ==========================================
maestro_data = []
order_map = {
    "Ingreso de requerimiento": "1", "Admisibilidad": "2", "No admisible": "3",
    "Tramitación tipo 1": "4", "Tramitación tipo 2": "4", "Tramitación tipo 3": "4",
    "Revisión": "5", "Devolver a tramitación": "6", "Firma": "7", "Devuelve a revisión": "8"
}

res_agg = df_new_log.groupby("ESTADO")["RESPONSABLE"].unique().apply(lambda x: ", ".join(sorted(x))).reset_index()

for idx, row in res_agg.iterrows():
    estado = row["ESTADO"]
    maestro_data.append({
        "ESTADO": estado,
        "EST_ORDEN": f"{order_map[estado]}. {estado}",
        "RESPONSABLE": row["RESPONSABLE"]
    })

df_new_maestro = pd.DataFrame(maestro_data).sort_values("EST_ORDEN")
df_new_maestro.to_csv("nuevo_maestro_estados.csv", sep=";", index=False, encoding='utf-8-sig')

print("✅ ¡LISTO! Archivos creados exitosamente:")
print("   - nuevo_log_procesos.csv (5.530 registros)")
print("   - nuevo_maestro_estados.csv")
