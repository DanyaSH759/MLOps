import mlflow
import os

from airflow.models import DAG
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from typing import Any, Dict, Literal

import io
import pytz
import logging
import json
import pickle
import pandas as pd

from datetime import datetime, timedelta

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from airflow.utils.dates import days_ago
from airflow.models import Variable
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.operators.python_operator import PythonOperator

from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient


# логи
_LOG = logging.getLogger()
_LOG.addHandler(logging.StreamHandler())

BUCKET = Variable.get("S3_BUCKET")

DEFAULT_ARGS = {
    "owner": "Shulyak_Danila",
    "retry": 3,
    "retry_delay": timedelta(minutes=1),
}

dag = DAG(
    dag_id='Shulyak_Danila',
    schedule_interval="0 1 * * *",
    start_date=days_ago(2),
    catchup=False,
    tags=["mlops"],
    default_args=DEFAULT_ARGS)

model_names = ["random_forest", "linear_regression", "desicion_tree"]
models = dict(
    zip(model_names, [
        RandomForestRegressor(),
        LinearRegression(),
        DecisionTreeRegressor(),
    ]))

def configure_mlflow():
    for key in [
        "MLFLOW_TRACKING_URI",
        "AWS_ENDPOINT_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
    ]:
        os.environ[key] = Variable.get(key)


def init(**kwargs) -> Dict[str, Any]:

    # логирование
    m_name = kwargs['templates_dict']['m_name'] # получаем имя для логирования
    start = datetime.now(pytz.timezone("Europe/Moscow")).strftime("%A, %D, %H:%M")
    _LOG.info(f'Обучение {m_name}. Время запуска обучения: {start}')

    # импорт конфиго mlflow
    configure_mlflow()

    # создание эксперемента с проверкой на его наличие
    try:
        mlflow.create_experiment("danila_shulyak")
        exp_id = mlflow.set_experiment("danila_shulyak").experiment_id
    except:
        exp_id = mlflow.set_experiment("danila_shulyak").experiment_id

    # создаем родительский ран обучения и получаем его id
    with mlflow.start_run(run_name="shulyakds",
                        experiment_id = exp_id, description = "parent") as parent_run:
        
        # сохранение id родительского рана
        parent_run_id = parent_run.info.run_id

    # словарь для передачи данных
    main_metrics = {}

    # запись ифномации в словарь с шага init
    main_metrics["time_start_dag"] = start
    main_metrics["experiment_id"] = exp_id
    main_metrics["run_id"] = parent_run_id

    # передача контекста
    return main_metrics


def get_data(**kwargs) -> Dict[str, Any]:


    # получения словаря main_metrics
    ti = kwargs['ti']
    main_metrics = ti.xcom_pull(task_ids='init')

    # логирование
    m_name = kwargs['templates_dict']['m_name'] # получаем имя для создания папки с данными на s3 и логирования
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

    # запись ифномации в словарь с шага get_data
    main_metrics["load_data"] = start
    main_metrics["finish_load_data"] = end
    main_metrics["dataset_size"] = data.shape

    # передача контекста
    return main_metrics

def prepare_data(**kwargs) -> Dict[str, Any]:
    

    # логирование  
    m_name = kwargs['templates_dict']['m_name'] # получаем имя для скачивания папки с данными на s3 и логирования
    start = datetime.now(pytz.timezone("Europe/Moscow")).strftime("%H:%M")
    _LOG.info(f'Старт предобработки данных: {start}.')

    # получения словаря main_metrics
    ti = kwargs['ti']
    main_metrics = ti.xcom_pull(task_ids='get_data')

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
        X, y, test_size=0.3, random_state=12345)

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

    # запись ифномации в словарь с шага prepare_data
    main_metrics["start_pr_data"] = start
    main_metrics["end_pr_data"] = end
    main_metrics["Feateres"] = features
    main_metrics["Target"] = target

    # передача контекста
    return main_metrics


def train_model(**kwargs) -> Dict[str, Any]:
    

    # логирование
    m_name = kwargs['templates_dict']['m_name'] # получаем имя для скачивания папки с данными на s3 и логирования
    start = datetime.now(pytz.timezone("Europe/Moscow")).strftime("%H:%M")
    _LOG.info(f'Старт обучения модели {start}.')

    # получения словаря main_metrics
    ti = kwargs['ti']
    main_metrics = ti.xcom_pull(task_ids='prepare_data')

    # берем данные со словоря
    exp_id = main_metrics['experiment_id']
    run_id = main_metrics['run_id']
    features = main_metrics['Feateres']
    target = main_metrics['Target']

    # получаем имя модели для обучения
    model_name = kwargs['templates_dict']['model_name']

    # Загрузка готовых датасетов с s3
    s3_hook = S3Hook("s3_connection")
    data = {}
    for name in ["X_train", "X_test", "y_train", "y_test"]:
        file = s3_hook.download_file(
            key=f"Shulyak_Danila/{m_name}/datasets/{name}.pkl",
            bucket_name=BUCKET,
        )
        data[name] = pd.read_pickle(file)


    #вернем формат датафрейма данным
    X_train_sc = pd.DataFrame(data["X_train"])
    X_test_sc = pd.DataFrame(data["X_test"])
    y_train_sc  = pd.Series(data["y_train"])
    y_test_sc = pd.Series(data["y_test"])

    #для сохранения датасета через infer_signature скачаем оригинальный датасет и
    #для корректной работу mlflow.evaluate добавим оригинальный датасет без скалирования
    #обучение и предсказывание модели будет на наших скалированных ранее данных
    file = s3_hook.download_file(
            key=f"Shulyak_Danila/{m_name}/datasets/california_housing.pkl",
            bucket_name=BUCKET,
        )
    data_or = pd.DataFrame(pd.read_pickle(file))
    X_train, X_test, y_train, y_test = train_test_split(data_or[features], data_or[target])
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.15)

    #загрузка конфигов mlflow
    configure_mlflow()

    # запуск детского рана
    with mlflow.start_run(run_name=model_name, experiment_id=exp_id,
                        nested=True, parent_run_id = run_id) as child_run:

        # создание - обучение модели
        model = models[model_name]
        model.fit(pd.DataFrame(X_train_sc), y_train_sc)
        prediction = model.predict(pd.DataFrame(X_test_sc))
        
        #валидационный датасет.
        eval_df = X_val.copy()
        eval_df["target"] = y_val

        #Сохранение результатов
        signature = infer_signature(data_or, prediction)
        model_info = mlflow.sklearn.log_model(model, model_name, signature=signature)
        mlflow.evaluate(
            model=model_info.model_uri,
            data=eval_df,
            targets="target",
            model_type="regressor",
            evaluators=["default"],
        )
    
    # т.к. это цикл то xcom тут не срабатывает и выдаёт null
    # метрики будут отдельно забираться в save_results


def save_results(**kwargs) -> None:
    

    #получаем имя для сохранения результата
    m_name = kwargs['templates_dict']['m_name']

    # собираем итоговый словарь который получился после 3 функций
    ti = kwargs['ti']
    all_results = ti.xcom_pull(task_ids='prepare_data')
    exp_id = all_results['experiment_id']
    run_id = all_results['run_id'] 

    # загрузка конфигов mlflow
    configure_mlflow()

    # получаем id детских ранов по id родительского рана
    client = mlflow.tracking.MlflowClient()
    all_runs = client.search_runs(experiment_ids=exp_id,
                                filter_string=f'tags.mlflow.parentRunId = "{run_id}"')
    child_runs = [run.info.run_id for run in all_runs]

    # теперь можем скачать метрики модели из mlflow
    result_metrics_models = {}
    for ch_run_id in child_runs:
        result_metrics_models[ch_run_id] = mlflow.get_run(ch_run_id).data.metrics

    # #создаём список со словорями всех логов
    all_results['all_metric'] = result_metrics_models

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

task_init = PythonOperator(task_id="init", python_callable=init, dag = dag,
                            templates_dict = {"m_name": '3_model_1_line'}, provide_context=True )

task_get_data = PythonOperator(task_id="get_data", python_callable=get_data, dag=dag,
                                templates_dict={"m_name": '3_model_1_line'}, provide_context=True )

task_prepare_data = PythonOperator(task_id="prepare_data", python_callable=prepare_data, dag=dag,
                                    templates_dict={"m_name": '3_model_1_line'}, provide_context=True )

task_save_results = PythonOperator(task_id="save_results", python_callable=save_results, dag=dag,
                                    templates_dict={"m_name": '3_model_1_line'}, provide_context=True )

# Для каждой модели создаем свою задачу
for model_name in model_names:
    training_model_tasks = PythonOperator(
        task_id=f'train_{model_name}',  #task_id для каждой модели
        python_callable=train_model,
        templates_dict={"m_name": '3_model_1_line', 'model_name': model_name}, 
        provide_context=True,
        dag=dag
    )

    task_init  >> task_get_data  >> task_prepare_data >> training_model_tasks >> task_save_results