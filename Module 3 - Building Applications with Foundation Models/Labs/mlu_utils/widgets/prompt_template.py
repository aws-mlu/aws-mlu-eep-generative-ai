import re
import ipywidgets as widgets
from IPython.display import display, Markdown, HTML
from langchain_core.prompts import PromptTemplate

def create_prompt_interface(bedrock_llm):
    # Apply minimal but effective styling
    display(HTML("""
    <style>
        .prompt-container { max-width: 900px; margin: 0 auto; }
        .section-header { color: #2c3e50; font-weight: 600; margin-top: 20px; 
                         margin-bottom: 12px; padding-bottom: 8px; border-bottom: 2px solid #ebf5fb; }
        .widget-button button { background-color: #3498db !important; color: white !important; 
                               border: none !important; border-radius: 4px !important; }
        .widget-button button:hover:enabled { background-color: #2980b9 !important; }
        .widget-button button:disabled { background-color: #bdc3c7 !important; opacity: 0.7 !important; }
        .prompt-display { background-color: #ecf0f1; border-left: 4px solid #3498db; 
                         padding: 12px; border-radius: 4px; font-family: monospace; margin: 10px 0; }
    </style>
    """))
    
    # Create UI components
    template_area = widgets.Textarea(
        value="Tell me a {content_type} about a {topic}.",
        placeholder="Enter your prompt template with {variables} in curly braces",
        layout=widgets.Layout(width='100%', height='100px')
    )
    
    variables_container = widgets.VBox([])
    
    output = widgets.Output(layout=widgets.Layout(
        margin='15px 0px', padding='15px', 
        border='1px solid #e0e0e0', border_radius='6px', min_height='100px'
    ))
    
    generate_button = widgets.Button(
        description="Generate Response",
        icon='paper-plane',
        disabled=True,
        layout=widgets.Layout(width='100%')
    )
    
    var_widgets = {}
    
    # Function to extract variables from template
    def extract_variables(template_string):
        return re.findall(r'\{([^{}]*)\}', template_string)
    
    # Update variable input fields based on template
    def update_variables(_=None):
        variables = extract_variables(template_area.value)
        variables_container.children = []
        var_widgets.clear()
        
        if not variables:
            generate_button.disabled = True
            return
        
        # Create input fields in a grid layout
        var_boxes = []
        for var in variables:
            var_input = widgets.Text(
                placeholder=f"Enter value for {var}",
                layout=widgets.Layout(width='100%')
            )
            var_widgets[var] = var_input
            var_input.observe(lambda _: validate_inputs(), names='value')
            
            var_box = widgets.VBox([
                widgets.HTML(f"<div style='font-weight:500'>{var}</div>"),
                var_input
            ], layout=widgets.Layout(margin='8px 0px', width='100%'))
            var_boxes.append(var_box)
        
        grid = widgets.GridBox(
            var_boxes,
            layout=widgets.Layout(
                grid_template_columns='repeat(auto-fill, minmax(200px, 1fr))',
                grid_gap='16px', width='100%'
            )
        )
        
        variables_container.children = [grid]
        validate_inputs()
    
    # Check if all inputs have values
    def validate_inputs():
        generate_button.disabled = not var_widgets or any(
            widget.value.strip() == "" for widget in var_widgets.values()
        )
    
    # Generate response from the LLM
    def generate_response(_):
        try:
            # Format the prompt with variable values
            template = PromptTemplate.from_template(template_area.value)
            values = {var: widget.value for var, widget in var_widgets.items()}
            formatted_prompt = template.format(**values)
            
            with output:
                output.clear_output()
                display(HTML("<div style='color:#3498db; font-weight:500;'>Generating response...</div>"))
                
                # Get response from LLM
                response = bedrock_llm.invoke(formatted_prompt).content
                
                # Display results
                output.clear_output()
                display(HTML("<div style='font-weight:500; margin-bottom:8px;'>Prompt:</div>"))
                display(HTML(f"<div class='prompt-display'>{formatted_prompt}</div>"))
                display(HTML("<div style='font-weight:500; margin:15px 0 8px 0; border-top:1px solid #e0e0e0; padding-top:15px;'>Response:</div>"))
                display(Markdown(response))
                
        except Exception as e:
            with output:
                output.clear_output()
                display(HTML(f"<div style='color:#e74c3c; font-weight:500;'>Error: {str(e)}</div>"))
    
    # Connect event handlers
    template_area.observe(update_variables, names='value')
    generate_button.on_click(generate_response)
    update_variables()  # Initial setup
    
    # Assemble the interface
    return widgets.VBox([
        widgets.HTML("<h2 class='section-header'>Interactive Prompt Template</h2>"),
        widgets.HTML("<div style='font-weight:500; margin:5px 0'>Template</div>"),
        template_area,
        widgets.HTML("<div style='color:#7f8c8d; font-size:0.9em; margin:5px 0;'>Use {variable_name} syntax to create input fields.</div>"),
        widgets.HTML("<h3 class='section-header'>Variables</h3>"),
        variables_container,
        widgets.HBox([generate_button], layout=widgets.Layout(display='flex', justify_content='flex-end', margin='20px 0 10px 0')),
        widgets.HTML("<h3 class='section-header'>Output</h3>"),
        output
    ], layout=widgets.Layout(
        padding='25px', border='1px solid #e0e0e0', 
        border_radius='8px', width='100%', 
        max_width='900px', margin='0 auto',
        class_='prompt-container'
    ))