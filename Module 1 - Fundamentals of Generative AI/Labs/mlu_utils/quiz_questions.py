import json

from random import randint
from typing import Dict, Any


class Quiz:
    def __init__(self, question):
        self.question_text = question["question"]
        self.options = question["options"]
        self.correct_index = question["correctIndex"]
        self.correct_answer = self.options[self.correct_index]
    
    def display(self):
        """Display this question as a simple but visually appealing widget"""
        import ipywidgets as widgets
        from IPython.display import display, clear_output, HTML
        
        # Create the question display with better styling
        question_display = widgets.HTML(
            value=f"""<div style="
                font-size: 18px; 
                font-weight: 500; 
                color: #2c3e50; 
                margin-bottom: 20px;
                padding: 15px;
                background-color: #f8f9fa;
                border-left: 4px solid #3498db;
                border-radius: 4px;
            ">{self.question_text}</div>"""
        )
        
        # We'll use a custom widget approach for better-looking options
        option_widgets = []
        
        # Create a container for selected option tracking
        selected_option = widgets.IntText(value=-1, layout=widgets.Layout(display='none'))
        
        # Create option buttons that look nicer than radio buttons
        for i, option in enumerate(self.options):
            # Create a button-like box for each option
            option_box = widgets.Button(
                description=option,
                layout=widgets.Layout(
                    width='100%',
                    height='auto',  # Auto height based on content
                    padding='12px 15px',
                    margin='8px 0',
                    border='2px solid #e9ecef',
                    border_radius='8px',
                )
            )
            
            # Add click handler to select this option
            def create_option_handler(idx):
                def option_handler(b):
                    # Update selected option
                    selected_option.value = idx
                    
                    # Update visual styling of all options
                    for j, opt in enumerate(option_widgets):
                        if j == idx:
                            # Selected option - blue background with dark text for better visibility
                            opt.button_style = ''  # Remove default button style
                            opt.style.button_color = '#e8f4fc'  # Light blue background
                            opt.style.text_color = '#0056b3'  # Dark blue text for contrast
                            opt.layout.border = '2px solid #3498db'  # Blue border
                        else:
                            # Unselected options
                            opt.button_style = ''
                            opt.style.button_color = '#ffffff'
                            opt.style.text_color = '#212529'  # Default dark text
                            opt.layout.border = '2px solid #e9ecef'
                    
                    # Enable submit button
                    submit_button.disabled = False
                
                return option_handler
            
            option_box.on_click(create_option_handler(i))
            option_widgets.append(option_box)
        
        # Create options container
        options_container = widgets.VBox(option_widgets)
        
        # Add CSS styling to fix text wrapping and button appearance
        css_styling = widgets.HTML("""
        <style>
            .widget-button {
                text-align: left !important;
                text-overflow: ellipsis;
                white-space: normal !important;
                word-wrap: break-word;
                overflow-wrap: break-word;
                font-weight: normal !important;
                box-shadow: none !important;
                transition: all 0.2s ease-in-out;
                min-height: 40px;
                height: auto !important;
                line-height: 1.5 !important;
            }
            .widget-button:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
            }
        </style>
        """)
        
        # Create submit button
        submit_button = widgets.Button(
            description='Submit Answer',
            button_style='primary',
            tooltip='Check your answer',
            layout=widgets.Layout(
                width='auto',
                margin='15px 0',
                font_weight='bold'
            ),
            disabled=True
        )
        
        # Create feedback area
        feedback = widgets.Output()
        
        # Function to handle submit button click
        def on_submit_click(b):
            selected_idx = selected_option.value
            
            if selected_idx < 0:
                return
                
            with feedback:
                clear_output()
                if selected_idx == self.correct_index:
                    display(HTML(
                        """<div style="
                            padding: 12px 15px;
                            background-color: #d4edda;
                            color: #155724;
                            border-radius: 5px;
                            margin-top: 15px;
                            font-weight: bold;
                            display: flex;
                            align-items: center;
                        ">
                            <span style="font-size: 24px; margin-right: 10px;">✓</span>
                            <span>Correct! Well done!</span>
                        </div>"""
                    ))
                    # Style the correct option
                    option_widgets[selected_idx].style.button_color = '#d4edda'
                    option_widgets[selected_idx].style.text_color = '#155724'
                else:
                    display(HTML(
                        f"""<div style="
                            padding: 12px 15px;
                            background-color: #f8d7da;
                            color: #721c24;
                            border-radius: 5px;
                            margin-top: 15px;
                            font-weight: bold;
                        ">
                            <span style="font-size: 24px; margin-right: 10px;">✗</span>
                            <span>Incorrect. The correct answer is: {self.correct_answer}</span>
                        </div>"""
                    ))
                    # Style the incorrect option
                    option_widgets[selected_idx].style.button_color = '#f8d7da'
                    option_widgets[selected_idx].style.text_color = '#721c24'
                    # Style the correct option
                    option_widgets[self.correct_index].style.button_color = '#d4edda'
                    option_widgets[self.correct_index].style.text_color = '#155724'
            
            # Disable all option buttons and submit button
            for opt in option_widgets:
                opt.disabled = True
            submit_button.disabled = True
            
            # Show the reset button
            reset_button.layout.display = 'block'
        
        # Attach click handler to submit button
        submit_button.on_click(on_submit_click)
        
        # Create reset button
        reset_button = widgets.Button(
            description='Try Again',
            button_style='warning',
            tooltip='Reset the question',
            layout=widgets.Layout(
                width='auto',
                margin='15px 0 15px 10px',
                display='none'  # Hidden initially
            )
        )
        
        # Function to reset the quiz
        def on_reset_click(b):
            # Reset selected option
            selected_option.value = -1
            
            # Reset option styling
            for opt in option_widgets:
                opt.disabled = False
                opt.button_style = ''
                opt.style.button_color = '#ffffff'
                opt.style.text_color = '#212529'  # Reset to default dark text
                opt.layout.border = '2px solid #e9ecef'  # Reset border
            
            # Reset submit button
            submit_button.disabled = True
            
            # Hide reset button
            reset_button.layout.display = 'none'
            
            # Clear feedback
            with feedback:
                clear_output()
        
        # Attach click handler to reset button
        reset_button.on_click(on_reset_click)
        
        # Create button container
        button_container = widgets.HBox([submit_button, reset_button])
        
        # First display the CSS styling
        display(css_styling)
        
        # Create the main container
        main_container = widgets.VBox([
            question_display,
            options_container,
            selected_option,  # Hidden tracker
            button_container,
            feedback
        ])
        
        # Add a nice border and shadow to the container
        main_container.layout.border = '1px solid #dee2e6'
        main_container.layout.padding = '20px'
        main_container.layout.margin = '15px 0'
        main_container.layout.border_radius = '10px'
        main_container.layout.box_shadow = '0 2px 10px rgba(0,0,0,0.1)'
        main_container.layout.max_width = '700px'  # Set a maximum width
        
        # Display the quiz widget
        display(main_container)


lab2a_question1 = Quiz(
    question={
        "question": "Which model parameter controls randomness or diversity in generated sequences?", "options" : [
            "temperature",
            "topP",
            "MaxTokens",
            "stopSequences",
        ],
        "correctIndex": 0,
    }
)


lab2b_question1 = Quiz(
    question={
        "question": "Where does a Strands Agent keep the memory of the conversation between calls?", "options" : [
            "The system_prompt",
            "The BedrockModel",
            "The agent.messages list",
            "A @tool function",
        ],
        "correctIndex": 2,
    }
)

lab3_question1 = Quiz(
    question={
        "question": "Which of the following is NOT a terminology used for of in-context learning techniques?", "options" : [
            "Few-shot learning",
            "Zero-shot learning",
            "One-shot learning",
            "Some-shot learning",
        ],
        "correctIndex": 3,
    }
)

lab3_question2 = Quiz(
    question={
        "question": "Chain-of-Thought (CoT) is most preferred for which of the following tasks?", "options" : [
            "Arithmetic Reasoning",
            "Summarization of essays",
            "Factual Q&A",
            "Machine translation",
        ],
        "correctIndex": 0,
    }
)

lab4a_question1 = Quiz(
    question={
        "question": "Which of the following outcomes is TRUE when temperature is set to zero?", "options" : [
            "You can use self-consistency technique.",
            "You can use the Tree-of-thought technique.",
            "You can use Chain-of-Thought technique.",
            "None of the above.",
        ],
        "correctIndex": 2,
    }
)

lab4b_question1 = Quiz(
    question={
        "question": "Which of the following techniques is expected to have the highest prompting cost?", "options" : [
            "Chain-of-thought (CoT)",
            "Self-consistency",
            "Tree-of-thought (ToT)",
            "Zero-shot prompting",
        ],
        "correctIndex": 2,
    }
)


lab5_question1 = Quiz(
    question={
        "question": "Which of the following is NOT a fully supported image format for Nova models?", "options" : [
            ".png",
            ".mp4",
            ".gif",
            ".jpg",
        ],
        "correctIndex": 2,
    }
)

lab5_question2 = Quiz(
    question={
        "question": "Which statement is true when natively prompting Nova models with GIFs.", "options":[
            "Only the first frame is considered",
            "GIFs are not supported by Nova.",
            "The entire sequence of frames are natively considered",
            "Only the last frame is considered.",
        ],
        "correctIndex": 0,
    }
)