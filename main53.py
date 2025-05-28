import pandas as pd
from gurobipy import Model, GRB
from gurobipy import quicksum

def cargar_datos():
    datos = {}
    return datos

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
    #alpha = data

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
