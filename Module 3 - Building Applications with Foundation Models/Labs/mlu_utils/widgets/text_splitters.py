import ipywidgets as widgets
from IPython.display import display, HTML
import random
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from langchain_core.documents import Document

def create_text_splitter_widget():
    # Apply minimal styling
    display(HTML("""
    <style>
        .container { max-width: 900px; margin: 0 auto; padding: 20px; }
        h2, h3 { color: #2c3e50; border-bottom: 2px solid #ebf5fb; padding-bottom: 8px; }
        .widget-button button { background-color: #3498db !important; color: white !important; }
        .output-area { border-left: 4px solid #3498db !important; }
    </style>
    """))
    
    # Default sample text
    default_text = """LLMs are a key artificial intelligence (AI) technology powering intelligent chatbots and other natural language processing (NLP) applications. The goal is to create bots that can answer user questions in various contexts by cross-referencing authoritative knowledge sources. Unfortunately, the nature of LLM technology introduces unpredictability in LLM responses. Additionally, LLM training data is static and introduces a cut-off date on the knowledge it has.
    
    Known challenges of LLMs include:
    
    - Presenting false information when it does not have the answer.
    - Presenting out-of-date or generic information when the user expects a specific, current response.
    - Creating a response from non-authoritative sources.
    - Creating inaccurate responses due to terminology confusion, wherein different training sources use the same terminology to talk about different things.
    
    You can think of the Large Language Model as an over-enthusiastic new employee who refuses to stay informed with current events but will always answer every question with absolute confidence. Unfortunately, such an attitude can negatively impact user trust and is not something you want your chatbots to emulate!
    
    RAG is one approach to solving some of these challenges. It redirects the LLM to retrieve relevant information from authoritative, pre-determined knowledge sources. Organizations have greater control over the generated text output, and users gain insights into how the LLM generates the response."""
    
    # Create the widgets
    text_input = widgets.Textarea(
        value=default_text,
        placeholder='Enter your text here...',
        description='Text:',
        disabled=False,
        layout=widgets.Layout(width='100%', height='200px')
    )
    
    splitter_type = widgets.Dropdown(
        options=['RecursiveCharacterTextSplitter', 'CharacterTextSplitter'],
        value='CharacterTextSplitter',
        description='Splitter:',
        disabled=False,
        layout=widgets.Layout(width='100%')
    )
    
    chunk_size = widgets.IntSlider(
        value=1200,
        min=100,
        max=5000,
        step=100,
        description='Chunk Size:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
        layout=widgets.Layout(width='100%')
    )
    
    chunk_overlap = widgets.IntSlider(
        value=60,
        min=0,
        max=500,
        step=10,
        description='Overlap:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
        layout=widgets.Layout(width='100%')
    )
    
    # Create a grid layout for checkboxes
    checkbox_layout = widgets.Layout(width='auto', margin='0 10px 0 0')
    separator_paragraph = widgets.Checkbox(value=True, description='Paragraph breaks (\\n\\n)', layout=checkbox_layout)
    separator_newline = widgets.Checkbox(value=True, description='New lines (\\n)', layout=checkbox_layout)
    separator_sentence = widgets.Checkbox(value=True, description='Sentence endings ((?<=\\. ))', layout=checkbox_layout)
    separator_space = widgets.Checkbox(value=True, description='Spaces ( )', layout=checkbox_layout)
    separator_empty = widgets.Checkbox(value=True, description='Empty string', layout=checkbox_layout)
    
    separator_custom = widgets.Text(
        value='',
        placeholder='Add custom separators (comma-separated)',
        description='Custom:',
        disabled=False,
        layout=widgets.Layout(width='100%')
    )
    
    # Group checkboxes in rows
    row1 = widgets.HBox([separator_paragraph, separator_newline, separator_sentence])
    row2 = widgets.HBox([separator_space, separator_empty])
    
    separators_box = widgets.VBox([
        widgets.HTML(value="<b>Separators (in order of priority):</b>"),
        row1,
        row2,
        separator_custom
    ], layout=widgets.Layout(margin='10px 0', padding='10px', border='1px solid #e0e0e0'))
    
    process_button = widgets.Button(
        description='Process Text',
        disabled=False,
        button_style='primary',
        tooltip='Click to split the text',
        icon='check',
        layout=widgets.Layout(width='auto')
    )
    
    output_area = widgets.Output(
        layout=widgets.Layout(
            border='1px solid #e0e0e0',
            padding='10px',
            margin='10px 0px',
            min_height='150px',
            class_='output-area'
        )
    )
    
    chunk_navigation = widgets.IntSlider(
        value=0,
        min=0,
        max=0,
        step=1,
        description='Chunk:',
        disabled=True,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
        layout=widgets.Layout(width='100%')
    )
    
    # Navigation buttons
    prev_button = widgets.Button(description='', disabled=True, icon='arrow-left', layout=widgets.Layout(width='40px'))
    next_button = widgets.Button(description='', disabled=True, icon='arrow-right', layout=widgets.Layout(width='40px'))
    chunk_info = widgets.HTML(value="No chunks available")
    
    # Global variable to store splits
    current_splits = []
    
    def update_ui(change):
        # Handle both dictionary and object style changes
        new_value = change.get('new') if isinstance(change, dict) else change.new
        
        if new_value == 'RecursiveCharacterTextSplitter':
            separators_box.layout.display = 'block'
            for widget in [separator_paragraph, separator_newline, separator_sentence, separator_space, separator_empty]:
                widget.value = True
        else:
            separators_box.layout.display = 'none'
    
    # Function to get selected separators
    def get_separators():
        separators = []
        if separator_paragraph.value:
            separators.append("\n\n")
        if separator_newline.value:
            separators.append("\n")
        if separator_sentence.value:
            separators.append("(?<=\\. )")
        if separator_space.value:
            separators.append(" ")
        if separator_empty.value:
            separators.append("")
        
        # Add custom separators if provided
        if separator_custom.value:
            custom_seps = [s.strip() for s in separator_custom.value.split(',')]
            separators.extend(custom_seps)
        
        return separators
    
    # Function to display a specific chunk
    def display_chunk(index):
        with output_area:
            output_area.clear_output()
            if 0 <= index < len(current_splits):
                chunk = current_splits[index]
                print(f"Chunk {index+1} of {len(current_splits)} (Length: {len(chunk.page_content)} characters):")
                print("-" * 50)
                print(chunk.page_content)
                print("-" * 50)
                
                # Update navigation buttons
                prev_button.disabled = index == 0
                next_button.disabled = index == len(current_splits) - 1
                chunk_info.value = f"Chunk {index+1} of {len(current_splits)}"
    
    # Function to process the text
    def process_text(b):
        nonlocal current_splits
        
        with output_area:
            output_area.clear_output()
            
            # Get the input text
            input_text = text_input.value
            if not input_text:
                print("Please enter some text to split.")
                return
            
            # Create a document
            doc = Document(page_content=input_text, metadata={})
            
            try:
                # Get the parameters
                selected_splitter = splitter_type.value
                size = chunk_size.value
                overlap = chunk_overlap.value
                
                # Process based on the selected splitter
                if selected_splitter == 'RecursiveCharacterTextSplitter':
                    separator_list = get_separators()
                    if not separator_list:
                        print("Warning: No separators selected. Using default empty string separator.")
                        separator_list = [""]
                    
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=size,
                        chunk_overlap=overlap,
                        separators=separator_list,
                        is_separator_regex=True,
                    )
                    print("Using RecursiveCharacterTextSplitter:")
                else:
                    print("Using CharacterTextSplitter:")
                    splitter = CharacterTextSplitter(
                        separator=" ",
                        chunk_size=size,
                        chunk_overlap=overlap,
                        length_function=len,
                    )
                
                # Split the document
                splits = splitter.split_documents([doc])
                
                # Store the splits for navigation
                current_splits = splits
                
                # Debug info
                print(f"Text split into {len(splits)} chunks")
                
                # Update the chunk navigation
                if splits:
                    chunk_navigation.max = len(splits) - 1
                    chunk_navigation.value = 0
                    chunk_navigation.disabled = False
                    prev_button.disabled = True
                    next_button.disabled = len(splits) <= 1
                    chunk_info.value = f"Chunk 1 of {len(splits)}"
                    
                    # Display the first chunk immediately
                    display_chunk(0)
                else:
                    print("No chunks were created. Try adjusting your parameters.")
                    
            except Exception as e:
                print(f"Error processing text: {str(e)}")
    
    # Function to handle chunk navigation
    def navigate_chunks(change):
        # Handle both dictionary and object style changes
        new_value = change.get('new') if isinstance(change, dict) else change.new
        display_chunk(new_value)
    
    # Functions for navigation buttons
    def go_prev(b):
        if chunk_navigation.value > 0:
            chunk_navigation.value -= 1
    
    def go_next(b):
        if chunk_navigation.value < chunk_navigation.max:
            chunk_navigation.value += 1
    
    # Connect the widgets to their functions
    splitter_type.observe(update_ui, names='value')
    process_button.on_click(process_text)
    chunk_navigation.observe(navigate_chunks, names='value')
    prev_button.on_click(go_prev)
    next_button.on_click(go_next)
    
    # Create the layout
    header = widgets.HTML(value="<h2>Text Splitter Demo</h2>")
    
    # Navigation controls
    nav_controls = widgets.HBox([
        prev_button, 
        chunk_info, 
        next_button
    ], layout=widgets.Layout(
        display='flex',
        align_items='center',
        justify_content='space-between',
        width='100%',
        margin='10px 0'
    ))
    
    # Button section with right alignment
    button_section = widgets.HBox([process_button], 
                                 layout=widgets.Layout(display='flex', justify_content='flex-end'))
    
    # Main layout
    main_layout = widgets.VBox([
        header,
        widgets.HTML(value="<h3>Input Text</h3>"),
        text_input,
        widgets.HTML(value="<h3>Splitter Parameters</h3>"),
        splitter_type,
        chunk_size,
        chunk_overlap,
        separators_box,
        button_section,
        widgets.HTML(value="<h3>Results</h3>"),
        chunk_navigation,
        nav_controls,
        output_area
    ], layout=widgets.Layout(
        padding='20px',
        border='1px solid #e0e0e0',
        border_radius='8px',
        width='100%',
        max_width='900px',
        margin='0 auto',
        class_='container'
    ))
    
    # Initialize UI based on default splitter type
    update_ui({'new': splitter_type.value})
    
    # Display the UI
    display(main_layout)