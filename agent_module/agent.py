from dotenv import load_dotenv
from crewai import LLM, Agent, Task, Crew, Process

load_dotenv()

llm = LLM(
    model = "gemini/gemini-2.0-flash",
    temperature = 0.7,
)

# Set your dataset path here
dataset_path = "../example_dataset/titanic.csv"  # manually set the path

# Define agents
eda_agent = Agent(role='EDA', goal='Explore data and report key stats.', backstory='Expert in EDA', verbose=True, llm=llm)
prep_agent = Agent(role='Preprocessor', goal='Clean and preprocess data.', backstory='Preprocessing specialist', verbose=True, llm=llm)
feat_agent = Agent(role='FeatureEngineer', goal='Engineer new features.', backstory='ML feature engineering expert', verbose=True, llm=llm)
train_agent = Agent(role='Trainer', goal='Train ML model & save code.', backstory='Model training veteran', verbose=True, llm=llm)
eval_agent = Agent(role='Evaluator', goal='Evaluate and summarize model.', backstory='Rigorous evaluator', verbose=True, llm=llm)

# Define tasks, saving code in /code and summary in /summary
eda_task = Task(
    description=f'Perform EDA on dataset at {dataset_path}; output Python code to /code/eda_code.py',
    agent=eda_agent,
    expected_output='EDA Python code',
    output_file='code/eda_code.py',
    inputs={'dataset_path': dataset_path}
)
eda_summary_task = Task(
    description=f'Summarize EDA findings for {dataset_path} in Markdown',
    agent=eda_agent,
    expected_output='EDA summary',
    output_file='summary/eda_summary.md',
    markdown=True,
    inputs={'dataset_path': dataset_path}
)

prep_task = Task(
    description=f'Preprocess data from {dataset_path} using EDA output; output Python code to /code/preprocess_code.py',
    agent=prep_agent,
    expected_output='Preprocessing Python code',
    output_file='code/preprocess_code.py',
    inputs={'dataset_path': dataset_path}
)
prep_summary_task = Task(
    description=f'Summarize preprocessing rationale for {dataset_path} in Markdown',
    agent=prep_agent,
    expected_output='Preprocessing summary',
    output_file='summary/preprocess_summary.md',
    markdown=True,
    inputs={'dataset_path': dataset_path}
)

feat_task = Task(
    description=f'Engineer features from preprocessed data ({dataset_path}); output Python code to /code/feature_code.py',
    agent=feat_agent,
    expected_output='Feature engineering Python code',
    output_file='code/feature_code.py',
    inputs={'dataset_path': dataset_path}
)
feat_summary_task = Task(
    description=f'Summarize feature engineering rationale for {dataset_path} in Markdown',
    agent=feat_agent,
    expected_output='Feature engineering summary',
    output_file='summary/feature_summary.md',
    markdown=True,
    inputs={'dataset_path': dataset_path}
)

train_task = Task(
    description=f'Train model on features from {dataset_path}, save as model.pkl; output Python code to /code/train_code.py',
    agent=train_agent,
    expected_output='Model training code and pickle',
    output_file='code/train_code.py',
    inputs={'dataset_path': dataset_path}
)
train_summary_task = Task(
    description=f'Summarize model training choices for {dataset_path} in Markdown',
    agent=train_agent,
    expected_output='Model training summary',
    output_file='summary/train_summary.md',
    markdown=True,
    inputs={'dataset_path': dataset_path}
)

eval_task = Task(
    description=f'Evaluate trained model (model.pkl) on {dataset_path}; output Python code to /code/eval_code.py',
    agent=eval_agent,
    expected_output='Model evaluation code',
    output_file='code/eval_code.py',
    inputs={'dataset_path': dataset_path}
)
eval_summary_task = Task(
    description=f'Summarize evaluation metrics for {dataset_path} in Markdown',
    agent=eval_agent,
    expected_output='Evaluation summary',
    output_file='summary/eval_summary.md',
    markdown=True,
    inputs={'dataset_path': dataset_path}
)

# Assemble crew for sequential execution
ml_crew = Crew(
    agents=[eda_agent, prep_agent, feat_agent, train_agent, eval_agent],
    tasks=[
        eda_task, eda_summary_task,
        prep_task, prep_summary_task,
        feat_task, feat_summary_task,
        train_task, train_summary_task,
        eval_task, eval_summary_task
    ],
    process=Process.sequential
)

# Run ML pipeline
result = ml_crew.kickoff()
print("Pipeline complete. All agent outputs saved in /code (Python files) and /summary (Markdown files).")
