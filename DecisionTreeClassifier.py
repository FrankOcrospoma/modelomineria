import numpy as np

class NodoDecision:
  def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
    self.feature_index = feature_index  # Índice de la característica utilizada para la división
    self.threshold = threshold          # Umbral de división para la característica
    self.left = left                    # Subárbol izquierdo (menor o igual al umbral)
    self.right = right                  # Subárbol derecho (mayor que el umbral)
    self.value = value                  # Valor de la hoja (solo se utiliza si el nodo es una hoja)

class ArbolDecision:
  def __init__(self, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1):
      self.criterion = criterion        # Criterio de división del árbol
      self.max_depth = max_depth        # Profundidad máxima del árbol
      self.min_samples_split = min_samples_split  # Número mínimo de muestras requeridas para dividir un nodo interno
      self.min_samples_leaf = min_samples_leaf    # Número mínimo de muestras requeridas para ser una hoja
      self.root = None                  # Nodo raíz del árbol

  def fit(self, X, y):
    self.root = self._construir_arbol(X, y, depth=0)

  def _construir_arbol(self, X, y, depth):
    num_samples, num_features = X.shape
    num_clases = len(np.unique(y))

    # Criterios de parada
    if depth == self.max_depth or num_clases == 1 or num_samples < self.min_samples_split:
      return NodoDecision(value=self._calcular_valor_nodo(y))

    # Encontrar la mejor división
    best_feature_index, best_threshold = self._encontrar_mejor_division(X, y)

    # Dividir los datos en función de la mejor división
    left_indices = np.where(X[:, best_feature_index] <= best_threshold)[0]
    right_indices = np.where(X[:, best_feature_index] > best_threshold)[0]

    left_subtree = self._construir_arbol(X[left_indices], y[left_indices], depth + 1)
    right_subtree = self._construir_arbol(X[right_indices], y[right_indices], depth + 1)

    return NodoDecision(best_feature_index, best_threshold, left_subtree, right_subtree)

  def _encontrar_mejor_division(self, X, y):
    num_samples, num_features = X.shape
    best_impurity = float('inf')
    best_feature_index, best_threshold = None, None

    for feature_index in range(num_features):
      thresholds = np.unique(X[:, feature_index])

      for threshold in thresholds:
        left_indices = np.where(X[:, feature_index] <= threshold)[0]
        right_indices = np.where(X[:, feature_index] > threshold)[0]
        impurity = self._calcular_impureza(y[left_indices], y[right_indices])

        if impurity < best_impurity:
          best_impurity = impurity
          best_feature_index = feature_index
          best_threshold = threshold

    return best_feature_index, best_threshold
    
  def _calcular_impureza(self, y_left, y_right):
    n_left, n_right = len(y_left), len(y_right)
    n_total = n_left + n_right

    p_left = np.sum(y_left != y_left[0]) / n_left if n_left > 0 else 0
    p_right = np.sum(y_right != y_right[0]) / n_right if n_right > 0 else 0

    impurity_left = self._calcular_impureza_nodo(y_left)
    impurity_right = self._calcular_impureza_nodo(y_right)

    impurity = (n_left / n_total) * impurity_left + (n_right / n_total) * impurity_right
    return impurity

  def _calcular_impureza_nodo(self, y):
    if self.criterion == 'gini':
      return 1 - np.sum((np.bincount(y) / len(y))**2)
    elif self.criterion == 'entropy':
      p = np.clip(np.bincount(y) / len(y), 1e-15, 1 - 1e-15)  # evita divisiones por cero
      return -np.sum(p * np.log2(p))
    else:
      raise ValueError("Criterio de impureza no válido")

  def _calcular_valor_nodo(self, y):
    return np.bincount(y).argmax()

  def predecir(self, X):
    return np.array([self._predecir_muestra(x, self.root) for x in X])

  def _predecir_muestra(self, x, nodo):
    if nodo.value is not None:
      return nodo.value
    if x[nodo.feature_index] <= nodo.threshold:
      return self._predecir_muestra(x, nodo.left)
    else:
      return self._predecir_muestra(x, nodo.right)