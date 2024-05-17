import pandas as pd
import numpy as np
import random


#data_rd = data reducida
data_rd = pd.read_csv("Formato_Dataset-Observatorio_Riesgo_nuevos.csv", encoding='latin1')

missing_columns = ['Unnamed: 0','FECHA_CORTE', 'ANIO_SUPERV', 'MODALIDAD_APROVECHAMIENTO', 'UBIGEO',
       'DEPARTAMENTO', 'PROVINCIA', 'DISTRITO', 'TITULO_HABILITANTE',
       'TITULAR', 'NOMBRE_PMF', 'RESOLUCION_APRUEBA_PMF',
       'FECHA_APROBACION_PMF', 'NUM_INFORME_SUPERVISION', 'FECHA_SUPERVISION',
       'ARBOLES_INEXISTENTES', 'VOLUMEN_ILEGAL', 'VOLUMEN_LEGAL',
       'FECHA_INGRESO_OBSERVATORIO']

for i in missing_columns:
  data_rd = data_rd.drop(columns=[i])

columnas_categoricas = ["OBSERVATORIO"]
columnas_numericas   = ["AREA_TH","AREA_POA","NUM_ARBOLES_APROBADOS","CANT_ESPECIES_APROBADOS", "VOL_APROBADO", "VOL_MOVILIZADO", "ARBOLES_SUPERVISADOS"]


# LabelEncoder de los datos!
from sklearn.preprocessing import LabelEncoder
# Preprocesamiento con LabelEncoderfrom
for c in columnas_categoricas:
    print(str(c))
    le = LabelEncoder()
    le.fit(data_rd[str(c)])
    data_rd[str(c)]=le.transform(data_rd[str(c)])


#print(data_rd.describe())

# Datos atípicos
# Tratamiento de Outliers Univariados!

# Creamos una funcion para poder visualizar los percentiles  ---- Estadisticos de orden
def Cuantiles(lista):
  c = [0,1,5,10,20,30,40,50,60,70,80,90,92.5,95,97.5,99,100]
  matrix = pd.concat([pd.DataFrame(c),pd.DataFrame(np.percentile(lista.dropna(),c))],axis = 1)
  matrix.columns = ["Cuantil","Valor_Variable"]
  return(matrix) 


Cuantiles(data_rd["AREA_TH"]).transpose()
Cuantiles(data_rd["AREA_POA"]).transpose()
Cuantiles(data_rd["NUM_ARBOLES_APROBADOS"]).transpose()
Cuantiles(data_rd["VOL_APROBADO"]).transpose()
Cuantiles(data_rd["VOL_MOVILIZADO"]).transpose()

cuantil_1_area_th = np.percentile(data_rd["AREA_TH"],1)
cuantil_97_area_th = np.percentile(data_rd["AREA_TH"],92)

cuantil_1_area_poa = np.percentile(data_rd["AREA_POA"],1)
cuantil_97_area_poa = np.percentile(data_rd["AREA_POA"],97.5)

cuantil_1_num_arboles_aprobados = np.percentile(data_rd["NUM_ARBOLES_APROBADOS"],1)
cuantil_97_num_arboles_aprobados = np.percentile(data_rd["NUM_ARBOLES_APROBADOS"],97.5)

cuantil_1_num_vol_aprobado = np.percentile(data_rd["VOL_APROBADO"],1)
cuantil_97_num_vol_aprobado = np.percentile(data_rd["VOL_APROBADO"],95)

cuantil_1_num_vol_movilizado = np.percentile(data_rd["VOL_MOVILIZADO"],1)
cuantil_97_num_vol_movilizado = np.percentile(data_rd["VOL_MOVILIZADO"],97.5)


#print("cuantil_97_area_th: ", cuantil_97_area_th)
#print("cuantil_97_area_poa: ", cuantil_97_area_poa)
#print("cuantil_97_num_arboles_aprobados: ", cuantil_97_num_arboles_aprobados)
#print("cuantil_97_num_vol_aprobado: ", cuantil_97_num_vol_aprobado)
#print("cuantil_97_num_vol_movilizado: ", cuantil_97_num_vol_movilizado)


data_rd.loc[data_rd["AREA_TH"]<cuantil_1_area_th,"AREA_TH"] = cuantil_1_area_th
data_rd.loc[data_rd["AREA_TH"]>cuantil_97_area_th,"AREA_TH"] = cuantil_97_area_th
data_rd.loc[data_rd["AREA_POA"]<cuantil_1_area_poa,"AREA_POA"] = cuantil_1_area_poa
data_rd.loc[data_rd["AREA_POA"]>cuantil_97_area_poa,"AREA_POA"] = cuantil_97_area_poa
data_rd.loc[data_rd["NUM_ARBOLES_APROBADOS"]<cuantil_1_num_arboles_aprobados,"NUM_ARBOLES_APROBADOS"] = cuantil_1_num_arboles_aprobados
data_rd.loc[data_rd["NUM_ARBOLES_APROBADOS"]>cuantil_97_num_arboles_aprobados,"NUM_ARBOLES_APROBADOS"] = cuantil_97_num_arboles_aprobados
data_rd.loc[data_rd["VOL_APROBADO"]<cuantil_1_num_vol_aprobado,"VOL_APROBADO"] = cuantil_1_num_vol_aprobado
data_rd.loc[data_rd["VOL_APROBADO"]>cuantil_97_num_vol_aprobado,"VOL_APROBADO"] = cuantil_97_num_vol_aprobado
data_rd.loc[data_rd["VOL_MOVILIZADO"]<cuantil_1_num_vol_movilizado,"VOL_MOVILIZADO"] = cuantil_1_num_vol_movilizado
data_rd.loc[data_rd["VOL_MOVILIZADO"]>cuantil_97_num_vol_movilizado,"VOL_MOVILIZADO"] = cuantil_97_num_vol_movilizado

#print(data_rd.describe())

from DecisionTreeClassifier import ArbolDecision

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_rd.drop('OBSERVATORIO',axis=1),
                                                    data_rd.OBSERVATORIO,
                                                    test_size=0.33,
                                                    stratify=data_rd.OBSERVATORIO,
                                                    random_state=100)

print(X_train.shape)

# Crear una instancia del árbol de decisión
arbol = ArbolDecision(max_depth=4, min_samples_split=11, min_samples_leaf=25)

# Entrenar el árbol de decisión con los datos de entrenamiento
arbol.fit(X_train.values, y_train.values)

# Realizar predicciones con el árbol entrenado
predicciones = arbol.predecir(X_test.values)

# Evaluar el desempeño del modelo
exactitud = np.mean(predicciones == y_test)
print("Exactitud del modelo:", exactitud)