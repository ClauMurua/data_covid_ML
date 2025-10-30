"""
DAG de Apache Airflow para ejecutar el pipeline COVID-19 ML
Proyecto: Análisis predictivo de casos COVID-19 en Chile
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    'owner': 'data_science_team',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email': ['admin@covidml.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'covid_ml_pipeline',
    default_args=default_args,
    description='Pipeline completo de ML para predicción COVID-19',
    schedule_interval='@daily',  # Ejecutar diariamente
    catchup=False,
    tags=['covid', 'machine-learning', 'kedro'],
)

data_processing = BashOperator(
    task_id='data_processing',
    bash_command='cd /opt/airflow/project && kedro run --pipeline=dp',
    dag=dag,
)

# Task 2: Ejecutar pipeline de data science
data_science = BashOperator(
    task_id='data_science',
    bash_command='cd /opt/airflow/project && kedro run --pipeline=ds',
    dag=dag,
)

# Task 3: Ejecutar pipeline de reporting
reporting = BashOperator(
    task_id='reporting',
    bash_command='cd /opt/airflow/project && kedro run --pipeline=rp',
    dag=dag,
)

# Definir dependencias del flujo
data_processing >> data_science >> reporting