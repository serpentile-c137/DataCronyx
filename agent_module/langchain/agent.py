import os
from dotenv import load_dotenv
from pathlib import Path
from typing import TypedDict, Optional, Annotated
import json

from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Set your dataset path here
dataset_path = "../example_dataset/titanic.csv"  # manually set the path

# Initialize the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7
)

class MLAgentState(TypedDict):
    """State for the ML pipeline workflow"""
    dataset_path: str
    current_step: str
    eda_code: Optional[str]
    eda_summary: Optional[str]
    preprocessing_code: Optional[str]
    preprocessing_summary: Optional[str]
    feature_code: Optional[str]
    feature_summary: Optional[str]
    training_code: Optional[str]
    training_summary: Optional[str]
    evaluation_code: Optional[str]
    evaluation_summary: Optional[str]
    messages: Annotated[list, lambda x, y: x + y]

def create_output_directories():
    """Create output directories for code and summary files"""
    Path("code").mkdir(exist_ok=True)
    Path("summary").mkdir(exist_ok=True)

def save_to_file(content: str, filepath: str):
    """Save content to file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

def eda_node(state: MLAgentState) -> MLAgentState:
    """Perform EDA analysis and generate code"""
    
    # EDA Code Generation
    eda_prompt = ChatPromptTemplate.from_template("""
    You are an expert data scientist specializing in Exploratory Data Analysis (EDA).
    
    Task: Generate comprehensive Python code for EDA on the dataset at: {dataset_path}
    
    Requirements:
    1. Load the dataset using pandas
    2. Display basic information (shape, dtypes, info)
    3. Check for missing values and duplicates
    4. Generate statistical summaries
    5. Create visualizations (histograms, correlation matrix, boxplots)
    6. Identify outliers
    7. Analyze target variable distribution (if applicable)
    
    Generate ONLY the Python code, no explanations. Make sure the code is complete and executable.
    """)
    
    eda_code_chain = eda_prompt | llm | StrOutputParser()
    eda_code = eda_code_chain.invoke({"dataset_path": state["dataset_path"]})
    
    # Save EDA code
    save_to_file(eda_code, "code/eda_code.py")
    
    # EDA Summary Generation
    summary_prompt = ChatPromptTemplate.from_template("""
    You are an expert data scientist. Based on the EDA code for dataset: {dataset_path}
    
    Generate a comprehensive Markdown summary that would typically be found from running this EDA code.
    Include:
    1. Dataset overview and structure
    2. Data quality assessment
    3. Key statistical insights
    4. Patterns and correlations discovered
    5. Recommendations for preprocessing
    
    Write in Markdown format with proper headers and formatting.
    """)
    
    summary_chain = summary_prompt | llm | StrOutputParser()
    eda_summary = summary_chain.invoke({"dataset_path": state["dataset_path"]})
    
    # Save EDA summary
    save_to_file(eda_summary, "summary/eda_summary.md")
    
    return {
        **state,
        "eda_code": eda_code,
        "eda_summary": eda_summary,
        "current_step": "preprocessing",
        "messages": state["messages"] + [AIMessage(content=f"EDA completed for {state['dataset_path']}")]
    }

def preprocessing_node(state: MLAgentState) -> MLAgentState:
    """Perform data preprocessing"""
    
    prep_prompt = ChatPromptTemplate.from_template("""
    You are a data preprocessing specialist.
    
    Task: Generate Python code for data preprocessing based on the dataset: {dataset_path}
    
    Consider the EDA findings and implement:
    1. Handle missing values appropriately
    2. Remove or treat outliers
    3. Encode categorical variables
    4. Scale numerical features if needed
    5. Create train-test split
    6. Save preprocessed data
    
    Generate ONLY the Python code, no explanations. Make sure the code is complete and executable.
    Previous EDA insights should guide your preprocessing decisions.
    """)
    
    prep_code_chain = prep_prompt | llm | StrOutputParser()
    prep_code = prep_code_chain.invoke({"dataset_path": state["dataset_path"]})
    
    save_to_file(prep_code, "code/preprocess_code.py")
    
    # Preprocessing summary
    summary_prompt = ChatPromptTemplate.from_template("""
    Generate a Markdown summary explaining the preprocessing rationale for dataset: {dataset_path}
    
    Include:
    1. Preprocessing steps taken
    2. Rationale for each decision
    3. Impact on data quality
    4. Recommendations for next steps
    
    Write in Markdown format.
    """)
    
    summary_chain = summary_prompt | llm | StrOutputParser()
    prep_summary = summary_chain.invoke({"dataset_path": state["dataset_path"]})
    
    save_to_file(prep_summary, "summary/preprocess_summary.md")
    
    return {
        **state,
        "preprocessing_code": prep_code,
        "preprocessing_summary": prep_summary,
        "current_step": "feature_engineering",
        "messages": state["messages"] + [AIMessage(content="Data preprocessing completed")]
    }

def feature_engineering_node(state: MLAgentState) -> MLAgentState:
    """Perform feature engineering"""
    
    feat_prompt = ChatPromptTemplate.from_template("""
    You are a feature engineering expert.
    
    Task: Generate Python code for feature engineering on the preprocessed data from: {dataset_path}
    
    Implement:
    1. Create new features based on domain knowledge
    2. Feature interactions and polynomial features
    3. Feature selection techniques
    4. Dimensionality reduction if appropriate
    5. Feature importance analysis
    
    Generate ONLY the Python code, no explanations.
    """)
    
    feat_code_chain = feat_prompt | llm | StrOutputParser()
    feat_code = feat_code_chain.invoke({"dataset_path": state["dataset_path"]})
    
    save_to_file(feat_code, "code/feature_code.py")
    
    # Feature engineering summary
    summary_prompt = ChatPromptTemplate.from_template("""
    Generate a Markdown summary for feature engineering on dataset: {dataset_path}
    
    Include:
    1. New features created
    2. Feature selection rationale
    3. Expected impact on model performance
    4. Feature importance insights
    
    Write in Markdown format.
    """)
    
    summary_chain = summary_prompt | llm | StrOutputParser()
    feat_summary = summary_chain.invoke({"dataset_path": state["dataset_path"]})
    
    save_to_file(feat_summary, "summary/feature_summary.md")
    
    return {
        **state,
        "feature_code": feat_code,
        "feature_summary": feat_summary,
        "current_step": "training",
        "messages": state["messages"] + [AIMessage(content="Feature engineering completed")]
    }

def training_node(state: MLAgentState) -> MLAgentState:
    """Train machine learning models"""
    
    train_prompt = ChatPromptTemplate.from_template("""
    You are a machine learning model training expert.
    
    Task: Generate Python code to train ML models on the engineered features from: {dataset_path}
    
    Implement:
    1. Try multiple algorithms (RandomForest, XGBoost, LogisticRegression, etc.)
    2. Hyperparameter tuning with GridSearchCV or RandomizedSearchCV
    3. Cross-validation
    4. Model comparison
    5. Save the best model as 'model.pkl'
    6. Generate training metrics
    
    Generate ONLY the Python code, no explanations.
    """)
    
    train_code_chain = train_prompt | llm | StrOutputParser()
    train_code = train_code_chain.invoke({"dataset_path": state["dataset_path"]})
    
    save_to_file(train_code, "code/train_code.py")
    
    # Training summary
    summary_prompt = ChatPromptTemplate.from_template("""
    Generate a Markdown summary for model training on dataset: {dataset_path}
    
    Include:
    1. Models tested
    2. Hyperparameter tuning approach
    3. Model selection rationale
    4. Training performance metrics
    5. Best model characteristics
    
    Write in Markdown format.
    """)
    
    summary_chain = summary_prompt | llm | StrOutputParser()
    train_summary = summary_chain.invoke({"dataset_path": state["dataset_path"]})
    
    save_to_file(train_summary, "summary/train_summary.md")
    
    return {
        **state,
        "training_code": train_code,
        "training_summary": train_summary,
        "current_step": "evaluation",
        "messages": state["messages"] + [AIMessage(content="Model training completed")]
    }

def evaluation_node(state: MLAgentState) -> MLAgentState:
    """Evaluate the trained model"""
    
    eval_prompt = ChatPromptTemplate.from_template("""
    You are a model evaluation expert.
    
    Task: Generate Python code to evaluate the trained model (model.pkl) on dataset: {dataset_path}
    
    Implement:
    1. Load the saved model
    2. Generate predictions on test set
    3. Calculate comprehensive evaluation metrics
    4. Create confusion matrix and classification report (if classification)
    5. Generate ROC curve and AUC (if applicable)
    6. Feature importance visualization
    7. Model interpretation with SHAP values
    
    Generate ONLY the Python code, no explanations.
    """)
    
    eval_code_chain = eval_prompt | llm | StrOutputParser()
    eval_code = eval_code_chain.invoke({"dataset_path": state["dataset_path"]})
    
    save_to_file(eval_code, "code/eval_code.py")
    
    # Evaluation summary
    summary_prompt = ChatPromptTemplate.from_template("""
    Generate a Markdown summary for model evaluation on dataset: {dataset_path}
    
    Include:
    1. Evaluation metrics achieved
    2. Model performance analysis
    3. Feature importance insights
    4. Model strengths and weaknesses
    5. Recommendations for improvement
    6. Business impact assessment
    
    Write in Markdown format.
    """)
    
    summary_chain = summary_prompt | llm | StrOutputParser()
    eval_summary = summary_chain.invoke({"dataset_path": state["dataset_path"]})
    
    save_to_file(eval_summary, "summary/eval_summary.md")
    
    return {
        **state,
        "evaluation_code": eval_code,
        "evaluation_summary": eval_summary,
        "current_step": "completed",
        "messages": state["messages"] + [AIMessage(content="Model evaluation completed")]
    }

def build_ml_pipeline():
    """Build the ML pipeline using LangGraph"""
    
    # Create state graph
    workflow = StateGraph(MLAgentState)
    
    # Add nodes
    workflow.add_node("eda", eda_node)
    workflow.add_node("preprocessing", preprocessing_node)
    workflow.add_node("feature_engineering", feature_engineering_node)
    workflow.add_node("training", training_node)
    workflow.add_node("evaluation", evaluation_node)
    
    # Define the flow
    workflow.add_edge(START, "eda")
    workflow.add_edge("eda", "preprocessing")
    workflow.add_edge("preprocessing", "feature_engineering")
    workflow.add_edge("feature_engineering", "training")
    workflow.add_edge("training", "evaluation")
    workflow.add_edge("evaluation", END)
    
    # Compile the workflow
    return workflow.compile()

def main():
    """Main execution function"""
    print("Starting ML Pipeline with LangGraph...")
    
    # Create output directories
    create_output_directories()
    
    # Build the pipeline
    pipeline = build_ml_pipeline()
    
    # Initial state
    initial_state = {
        "dataset_path": dataset_path,
        "current_step": "eda",
        "eda_code": None,
        "eda_summary": None,
        "preprocessing_code": None,
        "preprocessing_summary": None,
        "feature_code": None,
        "feature_summary": None,
        "training_code": None,
        "training_summary": None,
        "evaluation_code": None,
        "evaluation_summary": None,
        "messages": [HumanMessage(content=f"Process dataset: {dataset_path}")]
    }
    
    print(f"Processing dataset: {dataset_path}")
    
    # Run the pipeline
    try:
        # Stream the execution to see progress
        for step_output in pipeline.stream(initial_state):
            for node_name, node_output in step_output.items():
                current_step = node_output.get("current_step", "unknown")
                print(f"✅ Completed: {node_name} -> Next: {current_step}")
        
        print("\n🎉 Pipeline complete! All agent outputs saved in:")
        print("  📁 /code (Python files)")
        print("  📁 /summary (Markdown files)")
        
        # List generated files
        code_files = list(Path("code").glob("*.py"))
        summary_files = list(Path("summary").glob("*.md"))
        
        print("\n📋 Generated files:")
        print("Code files:")
        for file in code_files:
            print(f"  - {file}")
        print("Summary files:")
        for file in summary_files:
            print(f"  - {file}")
            
    except Exception as e:
        print(f"❌ Pipeline failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
