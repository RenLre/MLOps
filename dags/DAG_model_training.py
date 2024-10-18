import sys
import os
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../model_training')))
from model_training_test import train_and_evaluate_model  
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../HPE')))
from HPE_test import search_hyperparameters 

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
    'dag_model_training',
    default_args=default_args,
    description='DAG to run Model Training',
    schedule_interval=timedelta(days=1),
    catchup=False
)

def run_hpe():
    """Run the hyperparameter estimation script."""
    search_hyperparameter()

def run_model_training():
    """Run the model training script."""
    train_and_evaluate_model()

# Task to run the HPE script
run_hpe_task = PythonOperator(
    task_id='run_hpe',
    python_callable=run_hpe,
    dag=dag
)

# Task to run the model training script
run_model_training_task = PythonOperator(
    task_id='run_model_training',
    python_callable=run_model_training,
    dag=dag
)

# Set task dependencies: run_hpe must finish before run_model_training
run_hpe_task >> run_model_training_task
