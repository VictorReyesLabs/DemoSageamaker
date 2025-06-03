
import argparse
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib # Para guardar el modelo

if __name__ == '__main__':
    print("Iniciando script de entrenamiento...")
    parser = argparse.ArgumentParser()

    # Sagemaker pasa la ruta a los datos de entrenamiento y prueba
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    # Sagemaker pasa la ruta donde debe guardar el modelo
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))

    args = parser.parse_args()

    # Cargar los datos de entrenamiento
    train_data = pd.read_csv(os.path.join(args.train, 'iris_train.csv'), header=None)
    train_features = train_data.iloc[:, :-1]
    train_labels = train_data.iloc[:, -1]

    # Cargar los datos de prueba
    test_data = pd.read_csv(os.path.join(args.test, 'iris_test.csv'), header=None)
    test_features = test_data.iloc[:, :-1]
    test_labels = test_data.iloc[:, -1]

    print("Datos cargados para el entrenamiento.")

    # Entrenar el modelo de Regresión Logística
    model = LogisticRegression(max_iter=200)
    model.fit(train_features, train_labels)
    print("Modelo de Regresión Logística entrenado.")

    # Evaluar el modelo
    predictions = model.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)
    print(f"Precisión del modelo: {accuracy:.4f}")

    # Guardar el modelo entrenado en la ruta especificada por SageMaker
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
    print(f"Modelo guardado en: {os.path.join(args.model_dir, 'model.joblib')}")
