import pandas as pd
from gurobipy import Model, GRB
from gurobipy import quicksum

def cargar_datos():
    # Comisarías por comuna
    df_C = pd.read_excel("ComisariasXComuna.xlsx")
    C  = df_C["id_comisaría"].tolist()
    S  = df_C["comuna"].unique().tolist()
    dc = df_C.set_index("id_comisaría")["min_carabineros"].to_dict()
    Cs = df_C.groupby("comuna")["id_comisaría"].apply(list).to_dict() 
            #si quiero la cantidad en ves de los id es .apply(len)

    # Franjas horarias
    df_F = pd.read_excel("FranjaHoraria.xlsx", header=None)
    F = df_F.iloc[:,0].tolist()

    # Días
    df_T = pd.read_excel("Dias.xlsx", header=None)
    T = df_T.iloc[:,0].tolist()

    # Sueldos de carabineros
    df_P = pd.read_excel("SueldoXCarabinero.xlsx")
    P = df_P["id_carabinero"].tolist()
    beta = df_P.set_index("id_carabinero")["sueldo"].to_dict()

    # Material complementario
    df_G = pd.read_excel("Materiales.xlsx")
    G = df_G["id_material"].tolist()
    psig = df_G.set_index("id_material")["efectividad"].to_dict()
    lambdag = df_G.set_index("id_material")["costo_fijo"].to_dict()

    # Vehículos
    df_B = pd.read_excel("CostoXVehiculo.xlsx")
    B = df_B["id_vehiculo"].tolist()
    gammab = df_B.set_index("id_vehiculo")["costo_fijo"].to_dict()
    kappab = df_B.set_index("id_vehiculo")["efectividad"].to_dict()
    deltab = df_B.set_index("id_vehiculo")["maximo_carabineros"].to_dict()

    # Parámetros sueltos
    df_param = pd.read_excel("ParametrosSueltos.xlsx")
    xi = df_param.loc[0, "efectividad_patullaje_sin_vehiculo"]
    epsilon = df_param.loc[0, "dist_maxima_desplazamiento"]
    alpha = df_param.loc[0, "presupuesto_inicial"]

    # Distancias entre comunas (matriz)
    df_rho = pd.read_excel("DistanciaEntreComunas.xlsx", index_col=None)
    rho = { (s1, s2): df_rho.at[s1, s2]
           for s1 in df_rho.index
           for s2 in df_rho.columns }

    # Índice de criminalidad por (comuna, franja, día)
    df_i = pd.read_excel("IndiceCrimenxComuna.xlsx")
    isft = { (row["Comuna"], row["Franja Horaria"], row["Día"]):
        row["Índice Criminalidad"]
        for _, row in df_i.iterrows() }

    # Mínimos de vehículos y material por comuna
    df_min = pd.read_excel("Comunas.xlsx")
    etas = df_min.set_index("Comuna")["Min_vehiculos"].to_dict()
    ms = df_min.set_index("Comuna")["Min_material"].to_dict()

    data = {
        "C": C,
        "S": S,
        "dc": dc,
        "Cs": Cs,
        "F": F,
        "T": T,
        "P": P,
        "B": B,
        "G": G,
        "gammab": gammab,
        "beta": beta,
        "isft": isft,
        "etas": etas,
        "lambdag": lambdag,
        "psig": psig,
        "kappab": kappab,
        "xi": xi,
        "rho": rho,
        "epsilon": epsilon,
        "deltab": deltab,
        "ms": ms,
        "alpha": alpha
    }
    return data


def construir_modelo(data):
    model = Model()
    model.setParam("TimeLimit", 60)
    #S = data
    #F = data
    #T = data
    #C = data
    #P = data
    #B = data
    #G = data
    #dc = data
    #alpha =    ºººº data

    x = model.addVars(P, C, T, F, vtype = GRB.BINARY, name = "x_pctf")
    w = model.addVars(S, B, T, F, vtype = GRB.BINARY, name = "w_sbtf")
    j = model.addVars(S, B, P, T, F, vtype = GRB.BINARY, name = "j_sbptf")
    l = model.addVars(S, P, T, F, vtype = GRB.BINARY, name = "l_sptf")
    u = model.addVars(S, P, T, F, vtype = GRB.BINARY, name = "u_sptf")
    a = model.addVars(G, S, T, vtype = GRB.BINARY, name = "a_gst")
    k = model.addVars(T, vtype = GRB.CONTINUOUS, name = "k_t")
    v = model.addVars(S, T, F, vtype = GRB.CONTINUOUS, name = "l_sptf")
    model.update()

    model.addConstrs((quicksum(k[30] == alpha - x[p][c][t][f] >= dc[c] for p in P) x[i] <= M * w[i] for i in I), name="R1")
    model.addConstrs((quicksum(x[p][c][t][f] >= dc[c] for p in P) for c in C for t in T for f in F), name="R2")
    model.addConstrs((w[i] + w[int(k)-1] <= 1 for i in I for k in Pi[i] if k != '' and pd.notna(k)), name="R3")
    model.addConstrs((x[i] <= M * w[i] for i in I), name="R4")
    model.addConstrs((quicksum(aij[i][j] * x[i] for i in I) <= bj[j] + y[j] for j in J), name="R5")
    model.addConstrs((w[i] + w[int(k)-1] <= 1 for i in I for k in Pi[i] if k != '' and pd.notna(k)), name="R6")
    model.addConstrs((x[i] <= M * w[i] for i in I), name="R7")
    model.addConstrs((quicksum(aij[i][j] * x[i] for i in I) <= bj[j] + y[j] for j in J), name="R8")
    model.addConstrs((w[i] + w[int(k)-1] <= 1 for i in I for k in Pi[i] if k != '' and pd.notna(k)), name="R9")
    model.addConstrs((x[i] <= M * w[i] for i in I), name="R10")
    model.addConstrs((quicksum(aij[i][j] * x[i] for i in I) <= bj[j] + y[j] for j in J), name="R11")
    model.addConstrs((w[i] + w[int(k)-1] <= 1 for i in I for k in Pi[i] if k != '' and pd.notna(k)), name="R12")
    model.addConstr(quicksum(y[j] * cj[j] for j in J) <= W, name="R13")
    model.addConstr(quicksum(w[i] for i in I) <= N, name="R14")

    model.setObjective(
    gp.quicksum(v[s, t, f] * i[s, t, f] for s in S for t in T for f in F),
    GRB.MAXIMIZE
)
    return model
    

def resolver_modelo(model):

    model.optimize()
    return model

def imprimir_resultados(model):
    if model.Status == GRB.OPTIMAL or model.Status == GRB.TIME_LIMIT:
        print("\n--- Resultados ---")
        print(f"Valor óptimo del beneficio neto: ${model.ObjVal:.2f} pesos\n")

        print("Cantidad producida de cada producto (xᵢ):")
        for var in model.getVars():
            if var.varName.startswith("x"):
                idx = int(var.varName.split('[')[-1].rstrip(']')) + 1
                print(f"  Producto {str(idx)}: {var.X:.2f} unidades")
    else:
        print("No se encontró una solución óptima.")

def main():
    data = cargar_datos()
    model = construir_modelo(data)
    resultado = resolver_modelo(model)
    imprimir_resultados(resultado)
    pass

if __name__ == "__main__":
    main()
