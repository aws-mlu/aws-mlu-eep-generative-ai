import ipywidgets as widgets
from IPython.display import display, HTML, Markdown
import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

def create_rag_comparison_ui(retriever, bedrock_llm):
    # Apply styling
    display(HTML("""
    <style>
        .comparison-container { max-width: 900px; margin: 0 auto; }
        .section-header { color: #2c3e50; font-weight: 600; margin-top: 20px; 
                         margin-bottom: 12px; padding-bottom: 8px; border-bottom: 2px solid #ebf5fb; }
        .widget-button button { background-color: #3498db !important; color: white !important; 
                               border: none !important; border-radius: 4px !important; }
        .widget-button button:hover:enabled { background-color: #2980b9 !important; }
        .widget-button button:disabled { background-color: #bdc3c7 !important; opacity: 0.7 !important; }
        .response-area { background-color: #f8f9fa; border-left: 4px solid #3498db; 
                        padding: 12px; border-radius: 4px; margin: 10px 0; }
        .info-text { color: #7f8c8d; font-size: 0.9em; margin: 5px 0; }
        .comparison-table th { background-color: #ebf5fb; color: #2c3e50; }
        .comparison-table td { vertical-align: top; padding: 12px !important; }
    </style>
    """))
    
    # Define templates
    qa_template = """Use the given context to answer the question. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Keep the answer as concise as possible. 

    Context: {context}

    Question: {question}
    Answer:
    """

    llm_template = """Answer the question below. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Keep the answer as concise as possible.

    Question: {question}
    Answer:
    """
    
    # Create prompt templates
    qa_prompt_template = PromptTemplate.from_template(qa_template)
    llm_prompt_template = PromptTemplate.from_template(llm_template)
    
    # Define the chains
    retrieval_qa_chain = qa_prompt_template | bedrock_llm | StrOutputParser()
    llm_chain = llm_prompt_template | bedrock_llm | StrOutputParser()
    
    # Create widgets
    query_input = widgets.Textarea(
        value='How can you use Bedrock for multi-agent collaboration?',
        placeholder='Enter your question here...',
        description='',
        disabled=False,
        layout=widgets.Layout(width='100%', height='100px')
    )
    
    process_button = widgets.Button(
        description='Compare Responses',
        disabled=False,
        button_style='',
        tooltip='Generate responses from both RAG and LLM',
        icon='play',
        layout=widgets.Layout(width='auto')
    )
    
    output_area = widgets.Output(
        layout=widgets.Layout(
            border='1px solid #e0e0e0',
            padding='15px',
            margin='10px 0px',
            min_height='200px'
        )
    )
    
    status_indicator = widgets.HTML(value="")
    
    # Function to process the query
    def process_query(b):
        # Clear previous output
        output_area.clear_output()
        
        # Get the query
        query = query_input.value
        if not query:
            with output_area:
                display(HTML("<div style='color:#e74c3c; font-weight:500;'>Please enter a question.</div>"))
            return
        
        # Update status
        status_indicator.value = "<div style='color:#3498db; font-weight:500;'>Generating responses...</div>"
        process_button.disabled = True
        
        try:
            # First retrieve relevant documents
            with output_area:
                display(HTML("<div style='color:#7f8c8d;'>Retrieving relevant documents...</div>"))
            
            retrieved_docs = retriever.invoke(query)
            docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
            
            # Generate RAG response
            with output_area:
                display(HTML("<div style='color:#7f8c8d;'>Generating RAG response...</div>"))
            
            rag_response = retrieval_qa_chain.invoke({
                "question": query,
                "context": docs_content
            })
            
            # Generate LLM response
            with output_area:
                display(HTML("<div style='color:#7f8c8d;'>Generating LLM response...</div>"))
            
            llm_response = llm_chain.invoke({'question': query})
            
            # Display results
            with output_area:
                output_area.clear_output()
                
                # Show the query
                display(HTML(f"<div style='font-weight:500; margin-bottom:15px;'>Query: <span style='color:#2c3e50;'>{query}</span></div>"))
                
                # Create comparison table
                html_table = """
                <table class='comparison-table' style='width:100%; border-collapse:collapse; border:1px solid #e0e0e0;'>
                    <tr>
                        <th style='width:50%; padding:10px; text-align:left; border:1px solid #e0e0e0;'>RAG Response</th>
                        <th style='width:50%; padding:10px; text-align:left; border:1px solid #e0e0e0;'>Vanilla LLM Response</th>
                    </tr>
                    <tr>
                        <td style='border:1px solid #e0e0e0; padding:10px;'>{}</td>
                        <td style='border:1px solid #e0e0e0; padding:10px;'>{}</td>
                    </tr>
                </table>
                """.format(rag_response.replace('\n', '<br>'), llm_response.replace('\n', '<br>'))
                display(HTML(html_table))
                
                # Show retrieved documents
                display(HTML("<div style='font-weight:500; margin:20px 0 10px 0;'>Retrieved Documents:</div>"))
                for i, doc in enumerate(retrieved_docs):
                    display(HTML("""
                    <div style='margin-bottom:10px; padding:10px; background-color:#f8f9fa; border:1px solid #e0e0e0; border-radius:4px;'>
                        <div style='font-weight:500; color:#3498db; margin-bottom:5px;'>Document {}</div>
                        <div>{}</div>
                    </div>
                    """.format((i+1), doc.page_content.replace('\n', '<br>'))))
            
            # Reset status
            status_indicator.value = "<div style='color:#27ae60; font-weight:500;'>Responses generated successfully!</div>"
            
        except Exception as e:
            with output_area:
                output_area.clear_output()
                display(HTML(f"<div style='color:#e74c3c; font-weight:500;'>Error: {str(e)}</div>"))
            status_indicator.value = "<div style='color:#e74c3c; font-weight:500;'>An error occurred.</div>"
        
        finally:
            process_button.disabled = False
    
    # Connect button to function
    process_button.on_click(process_query)
    
    # Create the layout
    header = widgets.HTML(value="<h2 class='section-header'>RAG vs LLM-only Comparison</h2>")
    
    # Input section
    input_section = widgets.VBox([
        widgets.HTML(value="<div style='font-weight:500; margin:5px 0'>Question</div>"),
        query_input,
        widgets.HTML(value="<div class='info-text'>Enter a question to compare RAG and vanilla LLM responses.</div>")
    ])
    
    # Button section with right alignment
    button_section = widgets.HBox([
        status_indicator,
        process_button
    ], layout=widgets.Layout(
        display='flex',
        justify_content='space-between',
        align_items='center',
        margin='20px 0'
    ))
    
    # Main layout
    main_layout = widgets.VBox([
        header,
        widgets.HTML(value="<h3 class='section-header'>Input</h3>"),
        input_section,
        button_section,
        widgets.HTML(value="<h3 class='section-header'>Results</h3>"),
        output_area
    ], layout=widgets.Layout(
        padding='20px',
        border='1px solid #e0e0e0',
        border_radius='8px',
        width='100%',
        max_width='900px',
        margin='0 auto',
        class_='comparison-container'
    ))
    
    # Display the UI
    display(main_layout)