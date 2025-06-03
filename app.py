from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import numpy as np
import os
import psycopg2
from psycopg2 import pool
import uuid
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'new_secret_5678')

# Neon connection pool
db_pool = psycopg2.pool.SimpleConnectionPool(
    1, 20,
    host=os.environ.get('DB_HOST', 'ep-odd-boat-a5tpi1i2-pooler.us-east-2.aws.neon.tech'),
    port=os.environ.get('DB_PORT', 5432),
    database=os.environ.get('DB_NAME', 'neondb'),
    user=os.environ.get('DB_USER', 'neondb_owner'),
    password=os.environ.get('DB_PASSWORD', 'npg_zR21CxagGdVB'),
    sslmode='require'
)

# Load dataset
CSV_URL = "https://drive.google.com/uc?id=1oAR1BlvzLAZzYttNkNMjimPRIKeVgI6T"
df = pd.read_csv(CSV_URL)
for col in df.columns:
    if df[col].dtype == 'int64':
        df[col] = df[col].astype(int)

# Average values for display
AVERAGES = {
    "RevolvingUtilizationOfUnsecuredLines": "46%",
    "NumberOfTime30-59DaysPastDueNotWorse": 0.54,
    "DebtRatio": "43%",
    "MonthlyIncome": 6340
}

# Sample 15 rows (5 practice, 10 main tasks)
def sample_rows():
    sampled_rows = df.sample(15, random_state=np.random.randint(1000)).to_dict('records')
    for row in sampled_rows:
        for key, value in row.items():
            if isinstance(value, np.int64):
                row[key] = int(value)
            elif isinstance(value, np.float64):
                row[key] = float(value)
    np.random.shuffle(sampled_rows)
    return sampled_rows[:15]  # 5 practice + 10 main

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start():
    participant_name = request.form.get('participant_name', '').strip() or None
    participant_id = str(uuid.uuid4())
    session['participant_id'] = participant_id
    session['participant_name'] = participant_name
    session['tasks'] = sample_rows()
    session['practice_index'] = 0
    session['main_index'] = 0
    session['responses'] = []
    session['is_practice'] = True
    return redirect(url_for('task'))

@app.route('/task', methods=['GET', 'POST'])
def task():
    if 'participant_id' not in session:
        return redirect(url_for('index'))

    is_practice = session.get('is_practice', True)
    task_index = session['practice_index'] if is_practice else session['main_index']
    max_tasks = 5 if is_practice else 10
    task_data = session['tasks'][task_index] if task_index < len(session['tasks']) else None

    if task_index >= max_tasks:
        if is_practice:
            session['is_practice'] = False
            session['practice_index'] = 0
            session['main_index'] = 5  # Start main tasks from index 5
            return redirect(url_for('task'))
        else:
            # Save responses to database with retry logic
            for attempt in range(3):  # Retry up to 3 times
                conn = db_pool.getconn()
                try:
                    with conn.cursor() as cursor:
                        for response in session['responses']:
                            task_data = next(t for t in session['tasks'] if t['ID'] == response['ID'])
                            revolving_util = float(str(task_data.get('RevolvingUtilizationOfUnsecuredLines', '')).replace('%', '')) / 100 if '%' in str(task_data.get('RevolvingUtilizationOfUnsecuredLines', '')) else float(task_data.get('RevolvingUtilizationOfUnsecuredLines', 0))
                            late_payments = int(task_data.get('NumberOfTime30-59DaysPastDueNotWorse', 0))
                            debt_ratio = float(str(task_data.get('DebtRatio', '')).replace('%', '')) / 100 if '%' in str(task_data.get('DebtRatio', '')) else float(task_data.get('DebtRatio', 0))
                            monthly_income = float(str(task_data.get('MonthlyIncome', '')).replace('$', '').replace(',', '')) if any(c in str(task_data.get('MonthlyIncome', '')) for c in ['$', ',']) else float(task_data.get('MonthlyIncome', 0))
                            cursor.execute(
                                "INSERT INTO responses2 (participant_id, task_number, is_practice, initial_probability, ai_assisted_probability, uncertainty_assisted_probability, low_uncertainty_assisted_probability, actual_default, predicted_default_25k, predicted_default_50k, confidence_25k, confidence_50k, reward, revolving_utilization, late_payments_30_59, debt_ratio, monthly_income) "
                                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                                (session['participant_id'], response['task_number'], response['is_practice'], response['initial_probability'], response['ai_assisted_probability'], response['uncertainty_assisted_probability'], response['low_uncertainty_assisted_probability'], response['actual_default'], response['predicted_default_25k'], response['predicted_default_50k'], response['confidence_25k'], response['confidence_50k'], response['reward'], revolving_util, late_payments, debt_ratio, monthly_income)
                            )
                        conn.commit()
                        logger.info(f"Successfully saved {len(session['responses'])} responses for participant {session['participant_id']}")
                        break  # Exit retry loop on success
                except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
                    logger.error(f"Database error on attempt {attempt + 1}: {e}")
                    conn.rollback()
                    db_pool.putconn(conn, close=True)  # Close the faulty connection
                    if attempt == 2:  # Last attempt failed
                        logger.error("Failed to save responses after 3 attempts")
                        return "Database connection error. Please try again later.", 500
                    time.sleep(1)  # Wait before retrying
                except Exception as e:
                    logger.error(f"Unexpected error saving data: {e}")
                    conn.rollback()
                    db_pool.putconn(conn)
                    return f"Error saving data: {e}", 500
                finally:
                    if conn not in [None, False]:  # Ensure conn is valid before returning
                        db_pool.putconn(conn)
            return render_template('end.html')

    if request.method == 'POST':
        initial_probability = float(request.form.get('initial_probability', 0))
        if not (0 <= initial_probability <= 100):
            return "Invalid probability", 400
        session['current_initial_probability'] = initial_probability
        return redirect(url_for('step2'))

    customer_number = task_index + 1
    customer_info = {
        "RevolvingUtilizationOfUnsecuredLines": task_data["RevolvingUtilizationOfUnsecuredLines"],
        "NumberOfTime30-59DaysPastDueNotWorse": task_data["NumberOfTime30-59DaysPastDueNotWorse"],
        "DebtRatio": task_data["DebtRatio"],
        "MonthlyIncome": task_data["MonthlyIncome"]
    }
    return render_template('task.html', customer_info=customer_info, averages=AVERAGES, customer_number=customer_number, is_practice=is_practice)

@app.route('/step2', methods=['GET', 'POST'])
def step2():
    if 'participant_id' not in session or 'tasks' not in session:
        return redirect(url_for('index'))

    is_practice = session.get('is_practice', True)
    task_index = session['practice_index'] if is_practice else session['main_index']
    task_data = session['tasks'][task_index]
    initial_probability = session.get('current_initial_probability', 0)

    if request.method == 'POST':
        ai_assisted_probability = float(request.form.get('ai_assisted_probability', 0))
        if not (0 <= ai_assisted_probability <= 100):
            return "Invalid probability", 400
        session['current_ai_assisted_probability'] = ai_assisted_probability
        return redirect(url_for('step3'))

    # Simulate AI prediction (25k data points)
    predicted_default_25k = min(100.0, max(0.0, float(task_data.get('Confidence_Score', 0)) * 100))  # Cap at 0–100%
    decision_25k = "Reject" if predicted_default_25k > 50 else "Accept"
    return render_template('step2.html', initial_probability=initial_probability, predicted_default=predicted_default_25k, decision=decision_25k, training_size="25,000")

@app.route('/step3', methods=['GET', 'POST'])
def step3():
    if 'participant_id' not in session or 'tasks' not in session:
        return redirect(url_for('index'))

    is_practice = session.get('is_practice', True)
    task_index = session['practice_index'] if is_practice else session['main_index']
    task_data = session['tasks'][task_index]
    initial_probability = session.get('current_initial_probability', 0)
    ai_assisted_probability = session.get('current_ai_assisted_probability', 0)

    if request.method == 'POST':
        uncertainty_assisted_probability = float(request.form.get('uncertainty_assisted_probability', 0))
        if not (0 <= uncertainty_assisted_probability <= 100):
            return "Invalid probability", 400
        session['current_uncertainty_assisted_probability'] = uncertainty_assisted_probability
        return redirect(url_for('step4'))

    # Total uncertainty (probability of default)
    predicted_default_25k = min(100.0, max(0.0, float(task_data.get('Confidence_Score', 0)) * 100))  # Cap at 0–100%
    decision_25k = "Reject" if predicted_default_25k > 50 else "Accept"
    return render_template('step3.html', initial_probability=initial_probability, ai_assisted_probability=ai_assisted_probability, predicted_default=predicted_default_25k, decision=decision_25k)

@app.route('/step4', methods=['GET', 'POST'])
def step4():
    if 'participant_id' not in session or 'tasks' not in session:
        return redirect(url_for('index'))

    is_practice = session.get('is_practice', True)
    task_index = session['practice_index'] if is_practice else session['main_index']
    task_data = session['tasks'][task_index]
    initial_probability = session.get('current_initial_probability', 0)
    ai_assisted_probability = session.get('current_ai_assisted_probability', 0)
    uncertainty_assisted_probability = session.get('current_uncertainty_assisted_probability', 0)

    if request.method == 'POST':
        low_uncertainty_assisted_probability = float(request.form.get('low_uncertainty_assisted_probability', 0))
        if not (0 <= low_uncertainty_assisted_probability <= 100):
            return "Invalid probability", 400

        # Map Creditability string to numerical value
        creditability = task_data.get('Creditability', 'non-default').lower()
        creditability_value = 1.0 if creditability == 'default' else 0.0
        actual_default = creditability_value * 100  # Scale to 0 or 100

        # Calculate reward
        user_decision = "Reject" if low_uncertainty_assisted_probability > 50 else "Accept"
        actual_decision = "Reject" if actual_default > 50 else "Accept"
        reward = 2.50 if user_decision == actual_decision else 0.00

        # Store response
        session['responses'].append({
            'ID': task_data.get('ID', ''),
            'task_number': task_index + 1,
            'is_practice': session['is_practice'],
            'initial_probability': initial_probability,
            'ai_assisted_probability': ai_assisted_probability,
            'uncertainty_assisted_probability': uncertainty_assisted_probability,
            'low_uncertainty_assisted_probability': low_uncertainty_assisted_probability,
            'actual_default': actual_default,
            'predicted_default_25k': min(100.0, max(0.0, float(task_data.get('Confidence_Score', 0)) * 100)),
            'predicted_default_50k': min(100.0, max(0.0, float(task_data.get('Confidence_Score', 0)) * 90)),
            'confidence_25k': float(task_data.get('Confidence_Score', 0)),
            'confidence_50k': float(task_data.get('Confidence_Score', 0)) * 0.9,
            'reward': reward
        })

        # Update task index
        if session['is_practice']:
            session['practice_index'] += 1
        else:
            session['main_index'] += 1
        return redirect(url_for('task'))

    # Simulate AI prediction (50,000 data points, lower epistemic uncertainty)
    predicted_default_50k = min(100.0, max(0.0, float(task_data.get('Confidence_Score', 0)) * 90))  # Cap at 0–100%
    decision_50k = "Reject" if predicted_default_50k > 50 else "Accept"
    return render_template('step4.html', initial_probability=initial_probability, ai_assisted_probability=ai_assisted_probability, uncertainty_assisted_probability=uncertainty_assisted_probability, predicted_default=predicted_default_50k, decision=decision_50k, training_size="50,000")

@app.route('/test-db')
def test_db():
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT NOW();")
            result = cursor.fetchone()
            logger.info("Successfully tested database connection")
            return f"Database time: {result[0]}"
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return f"Connection failed: {e}"
    finally:
        if conn not in [None, False]:
            db_pool.putconn(conn)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))