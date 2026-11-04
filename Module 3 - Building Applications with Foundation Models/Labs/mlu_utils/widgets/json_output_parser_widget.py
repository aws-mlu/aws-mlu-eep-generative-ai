import ipywidgets as widgets
from IPython.display import display, HTML, Markdown
import json
from mlu_utils.strands_utils import ask_llm, parse_json_response

# Instructions inserted wherever the template contains {format_instructions}
FORMAT_INSTRUCTIONS = (
    "Return the output as a single valid JSON object. "
    "Do not include any explanation, comments, or Markdown code fences."
)

class JsonParserUI:
    def __init__(self, llm):
        """
        Initialize the JSON Parser UI
        
        Args:
            llm: Strands model (e.g. BedrockModel) to invoke
        """
        self.llm = llm
        self._apply_styling()
        self._create_widgets()
        self._setup_layout()
        
    def _apply_styling(self):
        """Apply custom CSS styling to the UI"""
        display(HTML("""
        <style>
            .json-parser-container { max-width: 900px; margin: 0 auto; }
            .section-header { 
                color: #2c3e50; 
                font-weight: 600; 
                margin-top: 20px; 
                margin-bottom: 12px; 
                padding-bottom: 8px; 
                border-bottom: 2px solid #ebf5fb; 
            }
            .widget-button button { 
                background-color: #3498db !important; 
                color: white !important; 
                border: none !important; 
                border-radius: 4px !important; 
                font-weight: 500 !important;
                padding: 8px 16px !important;
            }
            .widget-button button:hover:enabled { 
                background-color: #2980b9 !important; 
            }
            .widget-button button:disabled { 
                background-color: #bdc3c7 !important; 
                opacity: 0.7 !important; 
            }
            .json-result { 
                background-color: #f8f9fa; 
                border-left: 4px solid #3498db; 
                padding: 15px; 
                border-radius: 4px; 
                margin: 10px 0; 
                overflow: auto;
            }
            .json-key { 
                color: #2c3e50; 
                font-weight: bold; 
            }
            .json-value { 
                color: #0066cc; 
            }
            .json-list { 
                list-style-type: none; 
                padding-left: 20px; 
                margin: 5px 0;
            }
            .json-list-item { 
                list-style-type: disc; 
                margin: 5px 0;
            }
            .instruction-text {
                color: #7f8c8d; 
                font-size: 0.9em; 
                margin: 5px 0;
            }
            .output-container {
                border: 1px solid #e0e0e0;
                border-radius: 6px;
                padding: 15px;
                margin: 15px 0px;
                min-height: 100px;
                background-color: white;
            }
        </style>
        """))
        
    def _create_widgets(self):
        """Create all the UI widgets"""
        self.template_area = widgets.Textarea(
            value="""Generate a JSON text describing a fictional character with the following attributes:
- name
- age
- profession
- hobbies (as an array)
- address (as a nested object with street, city, and country)

{format_instructions}""",
            placeholder='Enter your template here...',
            layout=widgets.Layout(width='100%', height='180px')
        )

        self.generate_button = widgets.Button(
            description='Generate JSON',
            icon='paper-plane',
            button_style='primary',
            tooltip='Click to generate JSON from template',
            layout=widgets.Layout(width='200px')
        )
        self.generate_button.on_click(self._on_generate_button_clicked)

        self.output_area = widgets.Output(
            layout=widgets.Layout(
                width='100%',
                min_height='100px',
                max_height='300px',
                overflow='auto'
            )
        )
        
        self.json_view = widgets.HTML(value="")
        
    def _setup_layout(self):
        """Create the main layout"""
        self.main_box = widgets.VBox([
            widgets.HTML(value="<h2 class='section-header'>JSON Output Parsing Interactive Demo</h2>"),
            widgets.HTML(value="<div style='font-weight:500; margin:5px 0'>Template</div>"),
            self.template_area,
            widgets.HTML(value="<div class='instruction-text'>Use <code>{format_instructions}</code> where you want the JSON formatting instructions to appear.</div>"),
            widgets.HBox([self.generate_button], 
                        layout=widgets.Layout(
                            display='flex', 
                            justify_content='flex-end', 
                            margin='20px 0 10px 0'
                        )),
            widgets.HTML(value="<h3 class='section-header'>Output</h3>"),
            widgets.VBox([
                self.output_area,
                self.json_view
            ], layout=widgets.Layout(
                border='1px solid #e0e0e0',
                border_radius='6px',
                padding='15px',
                margin='10px 0px',
                min_height='150px',
                class_='output-container'
            ))
        ], layout=widgets.Layout(
            padding='25px', 
            border='1px solid #e0e0e0', 
            border_radius='8px', 
            width='100%', 
            max_width='900px', 
            margin='0 auto',
            class_='json-parser-container'
        ))
        
    def _format_json_for_display(self, json_obj):
        """Format JSON nicely for display"""
        html = "<h3 class='section-header'>Parsed JSON Result</h3>"
        html += "<div class='json-result'>"
        
        # Function to recursively build HTML for nested objects
        def build_html(obj, indent=0):
            result = ""
            if isinstance(obj, dict):
                result += "<ul class='json-list'>"
                for key, value in obj.items():
                    result += f"<li><span class='json-key'>{key}:</span> "
                    if isinstance(value, (dict, list)):
                        result += build_html(value, indent + 1)
                    else:
                        result += f"<span class='json-value'>{value}</span>"
                    result += "</li>"
                result += "</ul>"
            elif isinstance(obj, list):
                result += "<ul class='json-list'>"
                for item in obj:
                    result += "<li class='json-list-item'>"
                    if isinstance(item, (dict, list)):
                        result += build_html(item, indent + 1)
                    else:
                        result += f"<span class='json-value'>{item}</span>"
                    result += "</li>"
                result += "</ul>"
            return result
        
        html += build_html(json_obj)
        html += "</div>"
        return html
        
    def _on_generate_button_clicked(self, b):
        """Handle button click event"""
        with self.output_area:
            self.output_area.clear_output()
            display(HTML("<div style='color:#3498db; font-weight:500;'>Generating JSON from template...</div>"))
            
            try:
                # Fill the {format_instructions} placeholder with plain Python string formatting
                formatted_prompt = self.template_area.value.format(
                    format_instructions=FORMAT_INSTRUCTIONS
                )
                
                # Generate output with a Strands agent and parse the JSON text
                output = ask_llm(self.llm, formatted_prompt)
                result = parse_json_response(output)
                
                # Display the formatted JSON
                self.json_view.value = self._format_json_for_display(result)
                
                # Show the response
                self.output_area.clear_output()                
                display(HTML("<div style='font-weight:500; margin:15px 0 8px 0; border-top:1px solid #e0e0e0; padding-top:15px;'>Raw JSON:</div>"))
                display(HTML(f"<pre style='background-color: #ecf0f1; padding: 12px; border-radius: 4px; max-height: 150px; overflow: auto;'>{json.dumps(result, indent=2)}</pre>"))
                
            except Exception as e:
                self.output_area.clear_output()
                display(HTML(f"<div style='color:#e74c3c; font-weight:500;'>Error: {str(e)}</div>"))
                self.json_view.value = f"<div style='color: #e74c3c;'>Error: {str(e)}</div>"
    
    def display(self):
        """Display the UI"""
        display(self.main_box)