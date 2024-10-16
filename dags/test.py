# C:\Users\yrner\MLOPS\airflow\your-repo\dags\sample_dag.py

from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from datetime import datetime

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 10, 1),
    'retries': 1,
}

with DAG('test', default_args=default_args, schedule_interval='@daily') as dag:
    start = DummyOperator(task_id='start')
    end = DummyOperator(task_id='end')

    start >> end  # Define task dependencies
