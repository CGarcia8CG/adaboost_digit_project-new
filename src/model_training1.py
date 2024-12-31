import numpy as np
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
import joblib
from config import Config

# Cargar configuración
config = Config()
model_path = config.model_path_rfrl

def load_and_preprocess_data():
    """Carga y preprocesa los datos del conjunto digits."""
    digits = load_digits()
    X, y = digits.data, digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def optimize_hyperparameters(model, param_dist, X_train, y_train):
    """Optimiza los hiperparámetros del modelo usando RandomizedSearchCV."""
    search = RandomizedSearchCV(model, param_dist, n_iter=10, cv=3, random_state=42)
    search.fit(X_train, y_train)
    return search.best_estimator_

def train_adaboost_forest(X_train, y_train, weights):
    """Entrena un modelo AdaBoost con Random Forest."""
    # Optimizar hiperparámetros para Random Forest
    forest_params = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 10]}
    forest_model = RandomForestClassifier(random_state=42)
    best_forest = optimize_hyperparameters(forest_model, forest_params, X_train, y_train)

    adaboost_forest = AdaBoostClassifier(
        estimator=best_forest,
        n_estimators=50,
        learning_rate=1.0,
        random_state=42
    )
    adaboost_forest.fit(X_train, y_train, sample_weight=weights)

    # Guardar el modelo
    joblib.dump(adaboost_forest, model_path)
    return adaboost_forest

def evaluate_model(model, X_test, y_test):
    """Evalúa el modelo y muestra accuracy y F1-Score."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

def main():
    """Función principal para ejecutar el entrenamiento con Random Forest."""
    # Cargar datos
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    # Calcular pesos iniciales
    weights = compute_sample_weight('balanced', y_train)

    # Entrenar el modelo
    adaboost_forest = train_adaboost_forest(X_train, y_train, weights)

    # Evaluar el modelo
    print("Evaluación del modelo Random Forest con AdaBoost:")
    accuracy, report = evaluate_model(adaboost_forest, X_test, y_test)
    print("Accuracy:", accuracy)
    print("Classification Report:\n", report)

if __name__ == "__main__":
    main()
