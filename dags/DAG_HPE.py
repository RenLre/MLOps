from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import HPE

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 10, 16),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'DAG_HPE',
    default_args=default_args,
    description='DAG to run Hyperparameter Estimation',
    schedule_interval=timedelta(days=1),
    catchup=False
)

run_hpe_task = PythonOperator(
    task_id='run_hpe',
    python_callable=HPE.main,
    dag=dag
)

run_hpe_task