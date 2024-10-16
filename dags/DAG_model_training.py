from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.sensors.external_task_sensor import ExternalTaskSensor
from datetime import datetime, timedelta
import model_training

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
    'DAG_model_training',
    default_args=default_args,
    description='DAG to run Model Training',
    schedule_interval=timedelta(days=1),
    catchup=False
)

def fetch_hyperparameters():
    # Simulating fetching hyperparameters from a database
    return {
        'learning_rate': 0.01,
        'batch_size': 32,
        'epochs': 100
    }

def run_model_training(**kwargs):
    ti = kwargs['ti']
    hyperparameters = ti.xcom_pull(task_ids='fetch_hyperparameters')
    model_training.train_model(hyperparameters)

wait_for_hpe = ExternalTaskSensor(
    task_id='wait_for_hpe',
    external_dag_id='DAG_model_training.py',
    external_task_id='run_hpe',
    mode='reschedule',
    timeout=3600,
    dag=dag
)

fetch_hyperparameters_task = PythonOperator(
    task_id='fetch_hyperparameters',
    python_callable=fetch_hyperparameters,
    dag=dag
)

run_model_training_task = PythonOperator(
    task_id='run_model_training',
    python_callable=run_model_training,
    provide_context=True,
    dag=dag
)

wait_for_hpe >> fetch_hyperparameters_task >> run_model_training_task