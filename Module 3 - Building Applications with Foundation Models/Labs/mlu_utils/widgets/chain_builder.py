import ipywidgets as widgets
from IPython.display import display, clear_output, Markdown, HTML
import pandas as pd
from dataclasses import dataclass
from mlu_utils.strands_utils import ask_llm
import warnings


@dataclass
class PromptStep:
    """A prompt template string plus the variables it uses."""
    template: str
    input_variables: list

# Assuming bedrock_llm is already defined in your environment
warnings.filterwarnings("ignore")

def create_sequential_chain_builder(llm):
    # Apply styling
    display(HTML("""
    <style>
        .chain-builder-container { max-width: 1000px; margin: 0 auto; }
        .section-header { color: #2c3e50; font-weight: 600; margin-top: 20px; 
                         margin-bottom: 12px; padding-bottom: 8px; border-bottom: 2px solid #ebf5fb; }
        .subsection-header { color: #34495e; font-weight: 500; margin-top: 15px; 
                           margin-bottom: 10px; padding-bottom: 5px; border-bottom: 1px solid #f2f2f2; }
        .widget-button button { background-color: #3498db !important; color: white !important; 
                               border: none !important; border-radius: 4px !important; }
        .danger-button button { background-color: #e74c3c !important; color: white !important; 
                               border: none !important; border-radius: 4px !important; }
        .info-button button { background-color: #2ecc71 !important; color: white !important; 
                             border: none !important; border-radius: 4px !important; }
        .widget-button button:hover:enabled { background-color: #2980b9 !important; }
        .danger-button button:hover:enabled { background-color: #c0392b !important; }
        .info-button button:hover:enabled { background-color: #27ae60 !important; }
        .widget-button button:disabled { background-color: #bdc3c7 !important; opacity: 0.7 !important; }
        .danger-button button:disabled { background-color: #e6b0aa !important; opacity: 0.7 !important; }
        .info-button button:disabled { background-color: #abebc6 !important; opacity: 0.7 !important; }
        .chain-display { background-color: #ecf0f1; border-left: 4px solid #3498db; 
                        padding: 12px; border-radius: 4px; margin: 10px 0; }
        .var-display { background-color: #f8f9f9; border-left: 4px solid #2ecc71; 
                      padding: 10px; border-radius: 4px; margin: 8px 0; }
        .output-container { background-color: #f9f9f9; border: 1px solid #e0e0e0; 
                           border-radius: 6px; padding: 15px; margin-top: 15px; }
        .help-text { color: #7f8c8d; font-size: 0.9em; margin: 5px 0; }
        .step-container { border: 1px solid #e0e0e0; border-radius: 8px; 
                         padding: 20px; margin-bottom: 20px; background-color: #ffffff; }
        /* Fix for label width */
        .widget-label { min-width: 120px !important; max-width: 120px !important; }
        .textarea-label { min-width: 0px !important; max-width: 0px !important; }
    </style>
    """))
    
    class SequentialChainBuilder:
        def __init__(self, llm):
            self.llm = llm
            self.chains = []  # Will store chain step functions
            self.chain_names = []
            self.chain_prompts = []  # Store prompt templates
            self.chain_output_keys = []  # Store output keys
            self.input_variables = []
            self.available_variables = []
            self.output_variables = []
            
            # Main container
            self.main_container = widgets.VBox([])
            
            # Step 1: Input variables setup
            self.input_var_text = widgets.Text(
                description="Variable Name:",
                placeholder="e.g. title",
                layout=widgets.Layout(width='70%')
            )
            self.add_input_var_btn = widgets.Button(
                description="Add Variable",
                icon='plus',
                layout=widgets.Layout(width='28%')
            )
            self.add_input_var_btn.add_class('widget-button')
            self.add_input_var_btn.on_click(self.add_input_variable)
            
            self.input_vars_list = widgets.HTML(
                "<div class='help-text'>No input variables defined yet.</div>"
            )
            
            self.finalize_inputs_btn = widgets.Button(
                description="Finalize Input Variables",
                icon='check',
                layout=widgets.Layout(width='100%')
            )
            self.finalize_inputs_btn.add_class('info-button')
            self.finalize_inputs_btn.on_click(self.finalize_input_variables)
            
            # Input variable management
            self.input_var_dropdown = widgets.Dropdown(
                options=[],
                description='Select input:',
                disabled=True,
                layout=widgets.Layout(width='70%')
            )
            self.delete_input_var_btn = widgets.Button(
                description="Delete",
                icon='trash',
                disabled=True,
                layout=widgets.Layout(width='28%')
            )
            self.delete_input_var_btn.add_class('danger-button')
            self.delete_input_var_btn.on_click(self.delete_input_variable)
            
            # Step 2: Chain creation widgets
            self.chain_name = widgets.Text(
                description="Chain Name:",
                placeholder="e.g. screenwriter",
                layout=widgets.Layout(width='100%')
            )
            
            # Prompt template with proper sizing
            self.prompt_template = widgets.Textarea(
                placeholder="Enter your prompt template here. Use {variable_name} for variables.",
                layout=widgets.Layout(width='100%', height='150px')
            )
            
            self.output_key = widgets.Text(
                description="Output Key:",
                placeholder="e.g. synopsis",
                layout=widgets.Layout(width='70%')
            )
            self.available_vars_label = widgets.HTML(
                "<div class='var-display'>No variables available yet.</div>"
            )
            self.add_chain_btn = widgets.Button(
                description="Add Chain",
                icon='plus-square',
                disabled=True,
                layout=widgets.Layout(width='100%')
            )
            self.add_chain_btn.add_class('widget-button')
            self.add_chain_btn.on_click(self.add_chain)
            
            # Chain management
            self.chain_dropdown = widgets.Dropdown(
                options=[],
                description='Select chain:',
                disabled=True,
                layout=widgets.Layout(width='70%')
            )
            self.delete_chain_btn = widgets.Button(
                description="Delete",
                icon='trash',
                disabled=True,
                layout=widgets.Layout(width='28%')
            )
            self.delete_chain_btn.add_class('danger-button')
            self.delete_chain_btn.on_click(self.delete_chain)
            
            # Chain list display
            self.chain_list = widgets.HTML(
                "<div class='help-text'>No chains added yet.</div>"
            )
            
            # Step 3: Chain execution
            self.input_values = {}  # Will store widgets for each input variable
            self.input_values_container = widgets.VBox([])
            self.run_chain_btn = widgets.Button(
                description="Run Sequential Chain",
                icon='play',
                disabled=True,
                layout=widgets.Layout(width='100%')
            )
            self.run_chain_btn.add_class('info-button')
            self.run_chain_btn.on_click(self.run_chain)
            
            # Output area
            self.output_area = widgets.Output(
                layout=widgets.Layout(
                    margin='15px 0px',
                    min_height='200px'
                )
            )
            
            # Initialize the UI
            self.init_ui()
        
        def init_ui(self):
            # Step 1: Input Variables
            input_var_add_row = widgets.HBox([
                self.input_var_text,
                self.add_input_var_btn
            ], layout=widgets.Layout(width='100%'))
            
            input_var_management = widgets.HBox([
                self.input_var_dropdown,
                self.delete_input_var_btn
            ], layout=widgets.Layout(width='100%', margin='10px 0'))
            
            step1 = widgets.VBox([
                widgets.HTML("<h2 class='section-header'>Step 1: Define Input Variables</h2>"),
                widgets.HTML("<div class='help-text'>Define the input variables that will be used in your chain.</div>"),
                input_var_add_row,
                widgets.HTML("<h3 class='subsection-header'>Current Input Variables</h3>"),
                self.input_vars_list,
                input_var_management,
                self.finalize_inputs_btn
            ], layout=widgets.Layout(width='100%'))
            
            # Step 2: Chain Creation
            chain_management = widgets.HBox([
                self.chain_dropdown,
                self.delete_chain_btn
            ], layout=widgets.Layout(width='100%', margin='10px 0'))
            
            # Create a proper label for the prompt template
            prompt_label = widgets.HTML("<div style='font-weight:500; margin:15px 0 5px 0;'>Prompt Template</div>")
            
            step2 = widgets.VBox([
                widgets.HTML("<h2 class='section-header'>Step 2: Build Your Chain</h2>"),
                widgets.HTML("<div class='help-text'>Create chains that process your inputs and generate outputs.</div>"),
                self.available_vars_label,
                widgets.HTML("<div style='font-weight:500; margin:15px 0 5px 0;'>Chain Name</div>"),
                self.chain_name,
                prompt_label,
                self.prompt_template,
                widgets.HTML("<div class='help-text'>Use {variable_name} syntax to reference available variables.</div>"),
                widgets.HTML("<div style='font-weight:500; margin:15px 0 5px 0;'>Output Key</div>"),
                self.output_key,
                widgets.HTML("<div class='help-text'>This is the name of the variable that will store this chain's output.</div>"),
                widgets.VBox([self.add_chain_btn], layout=widgets.Layout(margin='15px 0')),
                widgets.HTML("<h3 class='subsection-header'>Manage Chains</h3>"),
                chain_management,
                widgets.HTML("<h3 class='subsection-header'>Current Chain Sequence</h3>"),
                self.chain_list
            ], layout=widgets.Layout(width='100%'))
            
            # Step 3: Chain Execution
            step3 = widgets.VBox([
                widgets.HTML("<h2 class='section-header'>Step 3: Execute Chain</h2>"),
                widgets.HTML("<div class='help-text'>Provide values for your input variables and run the sequential chain.</div>"),
                self.input_values_container,
                widgets.VBox([self.run_chain_btn], layout=widgets.Layout(margin='15px 0')),
                widgets.HTML("<h3 class='subsection-header'>Results</h3>"),
                widgets.VBox([self.output_area], layout=widgets.Layout(
                    border='1px solid #e0e0e0',
                    border_radius='6px',
                    padding='15px',
                    background_color='#f9f9f9'
                ))
            ], layout=widgets.Layout(width='100%'))
            
            # Wrap each step in a container
            step1_container = widgets.VBox([step1], layout=widgets.Layout(
                width='100%', 
                margin='0 0 20px 0'
            ))
            step1_container.add_class('step-container')
            
            step2_container = widgets.VBox([step2], layout=widgets.Layout(
                width='100%',
                margin='0 0 20px 0'
            ))
            step2_container.add_class('step-container')
            
            step3_container = widgets.VBox([step3], layout=widgets.Layout(
                width='100%'
            ))
            step3_container.add_class('step-container')
            
            # Main container
            self.main_container.children = [step1_container, step2_container, step3_container]
            self.main_container.layout = widgets.Layout(
                width='100%',
                max_width='1000px',
                margin='0 auto'
            )
            self.main_container.add_class('chain-builder-container')
            
            return self.main_container
        
        def add_input_variable(self, btn):
            input_var = self.input_var_text.value.strip()
            if not input_var:
                with self.output_area:
                    clear_output()
                    display(HTML("<div style='color:#e74c3c; font-weight:500;'>Please enter an input variable name.</div>"))
                return
            
            if input_var in self.input_variables:
                with self.output_area:
                    clear_output()
                    display(HTML(f"<div style='color:#e74c3c; font-weight:500;'>Input variable '{input_var}' already exists.</div>"))
                return
            
            self.input_variables.append(input_var)
            self.update_input_vars_list()
            self.input_var_text.value = ""
            
            with self.output_area:
                clear_output()
                display(HTML(f"<div style='color:#2ecc71; font-weight:500;'>Input variable '{input_var}' added successfully!</div>"))
        
        def delete_input_variable(self, btn):
            if not self.input_variables:
                return
            
            selected_idx = self.input_var_dropdown.value
            if selected_idx is None:
                with self.output_area:
                    clear_output()
                    display(HTML("<div style='color:#e74c3c; font-weight:500;'>Please select an input variable to delete.</div>"))
                return
            
            var_to_delete = self.input_variables[selected_idx]
            
            # Check if any chains depend on this input variable
            dependent_chains = []
            for i, prompt in enumerate(self.chain_prompts):
                if var_to_delete in prompt.input_variables:
                    dependent_chains.append(self.chain_names[i])
            
            if dependent_chains:
                with self.output_area:
                    clear_output()
                    error_msg = f"<div style='color:#e74c3c; font-weight:500;'>Cannot delete input variable '{var_to_delete}' because the following chains depend on it:</div><ul>"
                    for dep_chain in dependent_chains:
                        error_msg += f"<li>{dep_chain}</li>"
                    error_msg += "</ul><div style='color:#e74c3c;'>Please delete the dependent chains first.</div>"
                    display(HTML(error_msg))
                return
            
            # Remove the input variable
            self.input_variables.pop(selected_idx)
            if var_to_delete in self.available_variables:
                self.available_variables.remove(var_to_delete)
            
            # Update UI
            self.update_input_vars_list()
            self.update_available_vars_label()
            
            with self.output_area:
                clear_output()
                display(HTML(f"<div style='color:#2ecc71; font-weight:500;'>Input variable '{var_to_delete}' deleted successfully!</div>"))
        
        def update_input_vars_list(self):
            if not self.input_variables:
                self.input_vars_list.value = "<div class='help-text'>No input variables defined yet.</div>"
                self.input_var_dropdown.options = []
                self.input_var_dropdown.disabled = True
                self.delete_input_var_btn.disabled = True
                return
            
            var_list = "<div class='var-display'>"
            for var in self.input_variables:
                var_list += f"<div><code>{var}</code></div>"
            var_list += "</div>"
            self.input_vars_list.value = var_list
            
            # Update dropdown options
            self.input_var_dropdown.options = [(var, i) for i, var in enumerate(self.input_variables)]
            self.input_var_dropdown.disabled = False
            self.delete_input_var_btn.disabled = False
        
        def finalize_input_variables(self, btn):
            if not self.input_variables:
                with self.output_area:
                    clear_output()
                    display(HTML("<div style='color:#e74c3c; font-weight:500;'>Please add at least one input variable before finalizing.</div>"))
                return
            
            # Disable input variable management
            self.input_var_text.disabled = True
            self.add_input_var_btn.disabled = True
            self.input_var_dropdown.disabled = True
            self.delete_input_var_btn.disabled = True
            self.finalize_inputs_btn.disabled = True
            
            # Set available variables
            self.available_variables = self.input_variables.copy()
            self.update_available_vars_label()
            
            # Enable chain creation
            self.add_chain_btn.disabled = False
            
            # Create input fields for execution
            self.create_input_value_widgets()
            
            with self.output_area:
                clear_output()
                display(HTML("<div style='color:#2ecc71; font-weight:500;'>Input variables finalized. You can now start adding chains.</div>"))
        
        def create_input_value_widgets(self):
            widgets_list = []
            self.input_values = {}
            
            for var in self.input_variables:
                label = widgets.HTML(f"<div style='font-weight:500; margin:5px 0'>{var}</div>")
                self.input_values[var] = widgets.Text(
                    placeholder=f"Enter value for {var}",
                    layout=widgets.Layout(width='100%')
                )
                widgets_list.append(widgets.VBox([label, self.input_values[var]], 
                                                layout=widgets.Layout(margin='5px 0')))
            
            # Create a grid layout for input fields
            grid = widgets.GridBox(
                widgets_list,
                layout=widgets.Layout(
                    grid_template_columns='repeat(auto-fill, minmax(250px, 1fr))',
                    grid_gap='16px',
                    width='100%',
                    margin='10px 0'
                )
            )
            
            self.input_values_container.children = [grid]
        
        def update_available_vars_label(self):
            if not self.available_variables:
                self.available_vars_label.value = "<div class='var-display'>No variables available yet.</div>"
                return
                
            var_list = "<div class='var-display'><div style='font-weight:500; margin-bottom:5px;'>Available Variables:</div>"
            for var in self.available_variables:
                var_list += f"<code style='margin-right:10px; padding:3px 6px; background-color:#f0f0f0; border-radius:3px;'>{var}</code>"
            var_list += "</div>"
            self.available_vars_label.value = var_list
        
        def update_chain_list(self):
            if not self.chains:
                self.chain_list.value = "<div class='help-text'>No chains added yet.</div>"
                self.chain_dropdown.options = []
                self.chain_dropdown.disabled = True
                self.delete_chain_btn.disabled = True
                return
            
            chain_info = "<div class='chain-display'>"
            for i, (name, prompt, output_key) in enumerate(zip(self.chain_names, self.chain_prompts, self.chain_output_keys)):
                used_vars = ", ".join([f"<code style='padding:2px 4px; background-color:#f0f0f0; border-radius:3px;'>{var}</code>" for var in prompt.input_variables])
                chain_info += f"<div style='margin:8px 0; padding-bottom:12px; border-bottom:1px solid #d5dbdb;'>"
                chain_info += f"<div style='font-weight:500; color:#2c3e50; font-size:1.1em;'>{i+1}. {name}</div>"
                chain_info += f"<div style='margin:5px 0;'><span style='color:#7f8c8d; font-weight:500;'>Output:</span> <code style='padding:2px 4px; background-color:#f0f0f0; border-radius:3px;'>{output_key}</code></div>"
                chain_info += f"<div style='margin:5px 0;'><span style='color:#7f8c8d; font-weight:500;'>Uses:</span> {used_vars}</div>"
                
                # Show a preview of the prompt template
                template_preview = prompt.template
                if len(template_preview) > 100:
                    template_preview = template_preview[:100] + "..."
                chain_info += f"<div style='margin:5px 0; font-size:0.9em;'><span style='color:#7f8c8d; font-weight:500;'>Template:</span> <span style='font-family:monospace; color:#34495e;'>{template_preview}</span></div>"
                
                chain_info += "</div>"
            chain_info += "</div>"
            self.chain_list.value = chain_info
            
            # Update dropdown options
            self.chain_dropdown.options = [(f"{i+1}. {name}", i) for i, name in enumerate(self.chain_names)]
            self.chain_dropdown.disabled = False
            self.delete_chain_btn.disabled = False
        
        def add_chain(self, btn):
            name = self.chain_name.value.strip()
            template = self.prompt_template.value.strip()
            out_key = self.output_key.value.strip()
            
            if not (name and template and out_key):
                with self.output_area:
                    clear_output()
                    display(HTML("<div style='color:#e74c3c; font-weight:500;'>Please fill in all fields.</div>"))
                return
            
            # Check if output key already exists
            if out_key in self.available_variables:
                with self.output_area:
                    clear_output()
                    display(HTML(f"<div style='color:#e74c3c; font-weight:500;'>Output key '{out_key}' already exists. Please use a unique key.</div>"))
                return
            
            try:
                # Check which available variables are used in the template
                used_vars = [var for var in self.available_variables if "{" + var + "}" in template]
                
                if not used_vars:
                    with self.output_area:
                        clear_output()
                        display(HTML("<div style='color:#e74c3c; font-weight:500;'>Your prompt template must use at least one of the available variables.</div>"))
                    return
                
                prompt = PromptStep(template=template, input_variables=used_vars)
                
                # Each chain step is a plain Python function: format the prompt, call a Strands agent
                def chain(inputs, prompt=prompt):
                    return ask_llm(self.llm, prompt.template.format(**inputs))
                
                self.chains.append(chain)
                self.chain_names.append(name)
                self.chain_prompts.append(prompt)
                self.chain_output_keys.append(out_key)
                self.available_variables.append(out_key)
                self.output_variables.append(out_key)
                
                # Update UI
                self.update_available_vars_label()
                self.update_chain_list()
                self.run_chain_btn.disabled = False
                
                # Clear inputs for next chain
                self.chain_name.value = ""
                self.prompt_template.value = ""
                self.output_key.value = ""
                
                with self.output_area:
                    clear_output()
                    display(HTML(f"<div style='color:#2ecc71; font-weight:500;'>Chain '{name}' added successfully!</div>"))
                    
            except Exception as e:
                with self.output_area:
                    clear_output()
                    display(HTML(f"<div style='color:#e74c3c; font-weight:500;'>Error adding chain: {str(e)}</div>"))
        
        def delete_chain(self, btn):
            if not self.chains:
                return
            
            selected_idx = self.chain_dropdown.value
            if selected_idx is None:
                with self.output_area:
                    clear_output()
                    display(HTML("<div style='color:#e74c3c; font-weight:500;'>Please select a chain to delete.</div>"))
                return
            
            # Get the chain and its output key
            output_key_to_delete = self.chain_output_keys[selected_idx]
            name_to_delete = self.chain_names[selected_idx]
            
            # Check if any subsequent chains depend on this output key
            dependent_chains = []
            for i, prompt in enumerate(self.chain_prompts):
                if i != selected_idx and output_key_to_delete in prompt.input_variables:
                    dependent_chains.append(self.chain_names[i])
            
            if dependent_chains:
                with self.output_area:
                    clear_output()
                    error_msg = f"<div style='color:#e74c3c; font-weight:500;'>Cannot delete this chain because the following chains depend on its output '{output_key_to_delete}':</div><ul>"
                    for dep_chain in dependent_chains:
                        error_msg += f"<li>{dep_chain}</li>"
                    error_msg += "</ul><div style='color:#e74c3c;'>Please delete the dependent chains first.</div>"
                    display(HTML(error_msg))
                return
            
            # Remove the chain and its output key
            self.chains.pop(selected_idx)
            self.chain_names.pop(selected_idx)
            self.chain_prompts.pop(selected_idx)
            self.chain_output_keys.pop(selected_idx)
            self.output_variables.remove(output_key_to_delete)
            self.available_variables.remove(output_key_to_delete)
            
            # Update UI
            self.update_available_vars_label()
            self.update_chain_list()
            
            with self.output_area:
                clear_output()
                display(HTML(f"<div style='color:#2ecc71; font-weight:500;'>Chain '{name_to_delete}' deleted successfully!</div>"))
                
            # Disable run button if no chains left
            if not self.chains:
                self.run_chain_btn.disabled = True
        
        def run_chain(self, btn):
            if not self.chains:
                with self.output_area:
                    clear_output()
                    display(HTML("<div style='color:#e74c3c; font-weight:500;'>No chains to run.</div>"))
                return
            
            # Collect input values
            input_dict = {}
            missing_inputs = []
            
            for var in self.input_variables:
                value = self.input_values[var].value.strip()
                if value:
                    input_dict[var] = value
                else:
                    missing_inputs.append(var)
            
            if missing_inputs:
                with self.output_area:
                    clear_output()
                    error_msg = f"<div style='color:#e74c3c; font-weight:500;'>Please provide values for the following input variables:</div><ul>"
                    for var in missing_inputs:
                        error_msg += f"<li>{var}</li>"
                    error_msg += "</ul>"
                    display(HTML(error_msg))
                return
            
            # Validate chain dependencies before running
            with self.output_area:
                clear_output()
                display(HTML("<div style='color:#3498db; font-weight:500;'>Validating chain dependencies...</div>"))
                
                # Start with input variables
                available_vars = set(self.input_variables)
                
                # Check each chain in sequence
                for i, prompt in enumerate(self.chain_prompts):
                    chain_name = self.chain_names[i]
                    output_key = self.chain_output_keys[i]
                    required_vars = set(prompt.input_variables)
                    
                    # Check if all required variables are available
                    missing_vars = required_vars - available_vars
                    if missing_vars:
                        error_msg = f"<div style='color:#e74c3c; font-weight:500;'>Chain '{chain_name}' requires variables that are not available: {', '.join(missing_vars)}</div>"
                        error_msg += "<div style='color:#7f8c8d; margin-top:10px;'>Please check your prompt templates to ensure all variables are properly defined.</div>"
                        display(HTML(error_msg))
                        return
                    
                    # Add this chain's output to available variables for next chains
                    available_vars.add(output_key)
            
            try:
                with self.output_area:
                    clear_output()
                    display(HTML("<div style='color:#3498db; font-weight:500;'>Running sequential chain...</div>"))
                    
                    # Start with a dictionary containing the input variables
                    result = {k: input_dict[k] for k in self.input_variables}
                    
                    # Process each chain in sequence
                    for i, chain in enumerate(self.chains):
                        # Get the input variables for this chain
                        chain_inputs = self.chain_prompts[i].input_variables
                        output_key = self.chain_output_keys[i]
                        
                        # Extract only the inputs this chain needs
                        chain_input_dict = {k: result[k] for k in chain_inputs}
                        
                        # Run the chain and store its output
                        chain_output = chain(chain_input_dict)
                        result[output_key] = chain_output
                        
                        # Show progress
                        display(HTML(f"<div style='color:#3498db;'>Completed step {i+1}: {self.chain_names[i]}</div>"))
                    
                    # Display results in a DataFrame
                    display(HTML("<h3 style='color:#2c3e50; margin-top:15px;'>Results:</h3>"))
                    
                    # Create a styled HTML table for results
                    result_html = "<table style='width:100%; border-collapse:collapse; margin:15px 0;'>"
                    result_html += "<tr style='background-color:#3498db; color:white;'><th style='padding:8px; text-align:left;'>Variable</th><th style='padding:8px; text-align:left;'>Value</th></tr>"
                    
                    for i, (key, value) in enumerate(result.items()):
                        bg_color = "#f2f2f2" if i % 2 == 0 else "white"
                        result_html += f"<tr style='background-color:{bg_color};'>"
                        result_html += f"<td style='padding:8px; border-bottom:1px solid #ddd; font-weight:500;'>{key}</td>"
                        result_html += f"<td style='padding:8px; border-bottom:1px solid #ddd;'>{value}</td>"
                        result_html += "</tr>"
                    
                    result_html += "</table>"
                    display(HTML(result_html))
                                            
            except Exception as e:
                with self.output_area:
                    clear_output()
                    display(HTML(f"<div style='color:#e74c3c; font-weight:500;'>Error running chain: {str(e)}</div>"))
                    display(HTML("<div style='color:#7f8c8d; margin-top:10px;'>This might be due to an unexpected error in the chain execution.</div>"))
    
    # Create and return the builder instance
    builder = SequentialChainBuilder(llm)
    return builder.main_container