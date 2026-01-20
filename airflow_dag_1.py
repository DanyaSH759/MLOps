from airflow.models import DAG, Variable
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from typing import Any, Dict, Literal


import json
import pickle
import io
import logging
from datetime import datetime, timedelta
import pytz

import pandas as pd

from airflow.utils.dates import days_ago
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.models import DAG, Variable
from airflow.operators.python_operator import PythonOperator

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing


# логи
_LOG = logging.getLogger()
_LOG.addHandler(logging.StreamHandler())


# основные параметры
BUCKET = Variable.get("S3_BUCKET")

DEFAULT_ARGS = {
    "owner": "Shulyak_Danila",
    "retry": 3,
    "retry_delay": timedelta(minutes=1),
}

model_names = ["random_forest", "linear_regression", "desicion_tree"]

models = dict(
    zip(model_names, [
        RandomForestRegressor(),
        LinearRegression(),
        DecisionTreeRegressor(),
    ]))


def create_dag(dag_id: str, m_name: Literal["random_forest", "linear_regression", "desicion_tree"]):

    ####### DAG STEPS #######

    def init(**kwargs) -> Dict[str, Any]:

        

        # логирование
        start = datetime.now(pytz.timezone("Europe/Moscow")).strftime("%A, %D, %H:%M")
        _LOG.info(f'Обучение {m_name}. Время запуска обучения: {start}')

        # передача контекста
        return {"time": start,
                "model": m_name}
    
    def get_data(**kwargs) -> Dict[str, Any]:
        
        

        # логирование
        start = datetime.now(pytz.timezone("Europe/Moscow")).strftime("%A, %D, %H:%M")
        _LOG.info(f'Старт загрузки данных: {start}.')
        
        # Буду скачивать california_housing из sklearn
        housing = fetch_california_housing(as_frame=True)
        data = pd.concat([housing["data"], pd.DataFrame(housing["target"])], axis=1)
    
        # Загрузка датасета в папку
        s3_hook = S3Hook("s3_connection")
        filebuffer = io.BytesIO()
        data.to_pickle(filebuffer)
        filebuffer.seek(0)
        s3_hook.load_file_obj(
            file_obj=filebuffer,
            key=f'Shulyak_Danila/{m_name}/datasets/california_housing.pkl',
            bucket_name=BUCKET,
            replace=True,
        )
        
        # логирование
        end = datetime.now(pytz.timezone("Europe/Moscow")).strftime("%H:%M")
        _LOG.info(f'Конец загрузки данных: {end}.')
        _LOG.info(f'Размер датасета: строчки - {data.shape[0]}, колонки - {data.shape[-1]}.')

        # передача контекста
        return {"load_data": start,
                "finish_load_data": end,
               "dataset_size": data.shape}
    
    
    def prepare_data(**kwargs) -> Dict[str, Any]:
        

        # логирование
        start = datetime.now(pytz.timezone("Europe/Moscow")).strftime("%H:%M")
        _LOG.info(f'Старт предобработки данных: {start}.')

        # Скачка датасета с s3
        s3_hook = S3Hook("s3_connection")
        file = s3_hook.download_file(key=f"Shulyak_Danila/{m_name}/datasets/california_housing.pkl", bucket_name=BUCKET)
        data = pd.read_pickle(file)

        # т.к. колонка с таргетом известна, разделил датасет так
        features = [x for x in data.columns]
        features.remove('MedHouseVal')
        target = 'MedHouseVal'

        # Деление датасета
        X, y = data[features], data[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, random_state=12345)

        # скалирование
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)

   
        # сохранаяем на s3 все выборки
        s3_hook = S3Hook("s3_connection")
        for name, data in zip(
            ["X_train", "X_test", "y_train", "y_test"],
            [X_train_sc, X_test_sc, y_train, y_test],
        ):
            filebuffer = io.BytesIO()
            pickle.dump(data, filebuffer)
            filebuffer.seek(0)
            s3_hook.load_file_obj(
                file_obj=filebuffer,
                key=f"Shulyak_Danila/{m_name}/datasets/{name}.pkl",
                bucket_name=BUCKET,
                replace=True,
            )

        # логирование
        end = datetime.now(pytz.timezone("Europe/Moscow")).strftime("%H:%M")
        _LOG.info(f'Конец обработки данных: {end}.')
        _LOG.info(f'Фичи: {features}.')
        _LOG.info(f'таргет: {target}.')

        # передача контекста
        return {"Start": start,
                "End": end,
                "Feateres": features,
                "Target": target}


    def train_model(**kwargs) -> Dict[str, Any]:
        

        # логирование
        start = datetime.now(pytz.timezone("Europe/Moscow")).strftime("%H:%M")
        _LOG.info(f'Старт обучения модели {start}.')

        # Загрузка готовых датасетов с s3
        s3_hook = S3Hook("s3_connection")
        data = {}
        for name in ["X_train", "X_test", "y_train", "y_test"]:
            file = s3_hook.download_file(
                key=f"Shulyak_Danila/{m_name}/datasets/{name}.pkl",
                bucket_name=BUCKET,
            )
            data[name] = pd.read_pickle(file)

        #Обучение
        model = models[m_name]
        model.fit(data["X_train"], data["y_train"])
        prediction = model.predict(data["X_test"])

        #Просчет метрик
        result = {}
        result["r2_score"] = r2_score(data["y_test"], prediction)
        result["rmse"] = mean_squared_error(data["y_test"], prediction) ** 0.5
        result["mae"] = median_absolute_error(data["y_test"], prediction)

        #Логирование
        end = datetime.now(pytz.timezone("Europe/Moscow")).strftime("%H:%M")
        _LOG.info(f'Конец обучения модели {end}.')
        _LOG.info(f'Получившиеся метрики: {result}')

        # передача контекста
        return {"Start": start,
                "End": end,
                "metrics": result}

    def save_results(**kwargs) -> None:
        

        # Получаем конекст полученный со всех функций
        ti = kwargs['ti']
        
        # собираем в переменные
        result_task_1 = ti.xcom_pull(task_ids='init')
        result_task_2 = ti.xcom_pull(task_ids='get_data')
        result_task_3 = ti.xcom_pull(task_ids='prepare_data')
        result_task_4 = ti.xcom_pull(task_ids='train_model')

        #создаём список со словорями всех логов
        all_results = [result_task_1, result_task_2, result_task_3, result_task_4]

        # Сохраняем json со всеми логами на s3
        s3_hook = S3Hook("s3_connection")
        filebuffer = io.BytesIO()
        filebuffer.write(json.dumps(all_results, ensure_ascii=False, indent=4).encode('utf-8'))
        filebuffer.seek(0)
        s3_hook.load_file_obj(
            file_obj=filebuffer,
            key=f'Shulyak_Danila/{m_name}/results/metrics.json',
            bucket_name=BUCKET,
            replace=True,
        )

    # ####### INIT DAG #######

    dag = DAG(
    dag_id=dag_id,
    schedule_interval="0 1 * * *",
    start_date=days_ago(2),
    catchup=False,
    tags=["mlops"],
    default_args=DEFAULT_ARGS)

    with dag:
        # YOUR TASKS HERE
        task_init = PythonOperator(task_id="init", python_callable=init, dag = dag, templates_dict = {"m_name": m_name}, provide_context=True )

        task_get_data = PythonOperator(task_id="get_data", python_callable=get_data, dag=dag, templates_dict={"m_name": m_name}, provide_context=True )

        task_prepare_data = PythonOperator(task_id="prepare_data", python_callable=prepare_data, dag=dag, templates_dict={"m_name": m_name}, provide_context=True )

        task_train_model = PythonOperator(task_id="train_model", python_callable=train_model, dag=dag, templates_dict={"m_name": m_name}, provide_context=True )

        task_save_results = PythonOperator(task_id="save_results", python_callable=save_results, dag=dag, templates_dict={"m_name": m_name}, provide_context=True )

        task_init >> task_get_data >> task_prepare_data >> task_train_model >> task_save_results


for model_name in models.keys():
    create_dag(f"Danila_Shulyak_{model_name}", model_name)