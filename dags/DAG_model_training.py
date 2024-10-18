import sys
import os
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../model_training')))
from model_training_test import final_prediction 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../HPE')))
from HPE_test import search_hyperparameters 

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 10, 16),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'dag_model_training',
    default_args=default_args,
    description='DAG to run Model Training',
    schedule_interval=timedelta(days=1),
    catchup=False
)

def run_hpe(activation_function):
    """Run the hyperparameter estimation script."""
    search_hyperparameters(activation_function)

def run_model_training():
    """Run the model training script."""
    final_prediction()

# Task to run the HPE script for 'sigmoid'
run_hpe_sigmoid_task = PythonOperator(
    task_id='run_hpe_sigmoid',
    python_callable=run_hpe,
    op_kwargs={'activation_function': 'sigmoid'},
    dag=dag
)

# Task to run the HPE script for 'relu'
run_hpe_relu_task = PythonOperator(
    task_id='run_hpe_relu',
    python_callable=run_hpe,
    op_kwargs={'activation_function': 'relu'},
    dag=dag
)

# Task to run the model training script
run_model_training_task = PythonOperator(
    task_id='run_model_training',
    python_callable=run_model_training,
    dag=dag
)

# Set task dependencies: both HPE tasks must finish before model training
run_hpe_sigmoid_task >> run_model_training_task
run_hpe_relu_task >> run_model_training_task
