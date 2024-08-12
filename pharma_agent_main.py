import sqlite3
import pandas as pd
import json
import os
from dotenv import load_dotenv
import gradio as gr
from langchain_openai import OpenAI
from langchain.agents import create_sql_agent
from langchain.prompts import PromptTemplate
from langchain.sql_database import SQLDatabase
from sqlalchemy import create_engine

# Load environment variables from .env file
load_dotenv()

# Access the OpenAI API key from the environment
openai_api_key = os.getenv("OPENAI_API_KEY")

# Load CSV into SQLite Database using SQLAlchemy
def load_csv_to_sqlite(csv_file_path, db_name):
    engine = create_engine(f'sqlite:///{db_name}')
    df = pd.read_csv(csv_file_path)
    df.to_sql('data_table', engine, if_exists='replace', index=False)
    return engine

# Load Metadata
def load_metadata(metadata_file_path):
    with open(metadata_file_path, 'r') as f:
        metadata = json.load(f)
    return metadata

# Create the SQL Agent with Metadata
def create_agent_with_metadata(engine, metadata):
    schema = "\n".join([
        f"{field['field_name']} ({field['data_type']}): {field['definition']}"
        for field in metadata['fields']
    ])
    
    prompt_template = f"""
    You are an intelligent assistant that converts natural language queries into SQL queries.
    
    The data table schema is:
    {schema}
    
    Convert the following query to SQL:
    {{query}}
    
    SQL:
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["query"])
    
    llm = OpenAI(api_key=openai_api_key)
    db = SQLDatabase(engine)

    agent = create_sql_agent(
        llm=llm,
        db=db,
        verbose=True,
        top_k=10,
    )

    return agent

# Execute the Natural Language Query and return results
def execute_natural_language_query(agent, natural_language_query):

    res = agent.invoke({"input": natural_language_query})
    # res = agent.run(natural_language_query)
    print('this is res: ', res)
    print('this is the type of res: ', type(res))
    filtered = res['output']
    print('this is type of res output: ', type(filtered))
    print('this is the output: ', res['output'])
    # Assuming the result is a list of tuples
    return res['output']

# Function to create a UI element for each study
def create_study_ui_element(study):
    study_title = study[1]
    nct_number = study[0]
    study_status = study[3]
    study_url = study[2]
    
    html_content = f"""
    <div style="border:1px solid #ccc; padding: 10px; margin-bottom: 10px;">
        <h3>{study_title}</h3>
        <p><strong>NCT Number:</strong> {nct_number}</p>
        <p><strong>Status:</strong> {study_status}</p>
        <p style="text-align: right;"><a href="{study_url}" target="_blank">Study URL</a></p>
    </div>
    """
    
    return html_content

# Gradio Interface Function
def query_agent(natural_language_query):
    results = execute_natural_language_query(agent, natural_language_query)
    print('this is the type of the natural language query', type(natural_language_query))
    # Generate UI elements for each study
    ui_elements = []
    for study in results:
        print('this is the study:', study)
        print('this is the type of study: ', type(study))
        ui_elements.append(create_study_ui_element(study))
    
    # Combine all UI elements into one HTML output
    return "\n".join(ui_elements)

# Main Application Setup
def main():
    global engine, agent

    csv_file_path = 'data/clinical_trials.csv'
    metadata_file_path = 'data/clinical_trials_metadata.json' 
    db_name = 'data.db'

    engine = load_csv_to_sqlite(csv_file_path, db_name)
    metadata = load_metadata(metadata_file_path)
    agent = create_agent_with_metadata(engine, metadata)

    # Gradio Interface
    interface = gr.Interface(
        fn=query_agent,
        inputs="text",
        outputs="html",
        title="Clinical Trial",
        description="Enter a natural language query to interact with the SQL database."
    )

    interface.launch()

if __name__ == "__main__":
    main()