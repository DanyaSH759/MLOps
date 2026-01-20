# библиотеки
import mlflow
import os
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository


# модели и данные для обучения
models = dict(zip(["random_forest", "linear_regression", "desicion_tree"], 
                  [RandomForestRegressor(), LinearRegression(), DecisionTreeRegressor()]))
housing = fetch_california_housing(as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(housing['data'], housing['target'])
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.15)

# создание эксперемента с проверкой на его наличие
try:
    mlflow.create_experiment("danila_shulyak")
    exp_id = mlflow.set_experiment("danila_shulyak").experiment_id
except:
    exp_id = mlflow.set_experiment("danila_shulyak").experiment_id

with mlflow.start_run(run_name="shulyakds", experiment_id = exp_id, description = "parent") as parent_run:
    # запуск дочерних ранов с циклом по нашим моделям
    for model_name in models.keys():
        with mlflow.start_run(run_name=model_name, experiment_id=exp_id, nested=True) as child_run:
            # создание - обучение модели
            model = models[model_name]
            model.fit(pd.DataFrame(X_train), y_train)
            prediction = model.predict(X_val)
        
            #валидационный датасет.
            eval_df = X_val.copy()
            eval_df["target"] = y_val
        
            #Сохранение результатов
            signature = infer_signature(housing['data'], prediction)
            model_info = mlflow.sklearn.log_model(model, model_name, signature=signature)
            mlflow.evaluate(
                model=model_info.model_uri,
                data=eval_df,
                targets="target",
                model_type="regressor",
                evaluators=["default"],
            )
