import ipywidgets as widgets
from IPython.display import display, clear_output
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from mlu_utils.langchain_modules.memory_modules import ConversationBufferMemory
from langchain_aws import ChatBedrockConverse
import PyPDF2
import io

class SimpleChatbot:
    """
    A simple chatbot class using AWS Bedrock and LangChain.
    
    This class creates a user-friendly chatbot interface with:
    - System message customization
    - Temperature and max tokens controls
    - Conversation memory
    - Clean, responsive UI
    """
    
    def __init__(self, model_id="amazon.nova-pro-v1:0", 
                 default_system_message="You are a helpful, friendly AI assistant. Be concise and clear in your responses.",
                 default_temperature=0,
                 default_max_tokens=1000):
        """Initialize the chatbot with model and default parameters."""
        
        # Store default values
        self.model_id = model_id
        self.default_system_message = default_system_message
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        
        # Initialize LLM
        self.llm = ChatBedrockConverse(
            model=self.model_id,
            temperature=self.default_temperature,
            max_tokens=None if default_max_tokens == 0 else default_max_tokens
        )
        
        # Initialize memory
        self.memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
        
        # Create the prompt template and chain
        self._update_prompt_and_chain()
        
        # Create UI components
        self._create_ui_components()
        
        # Connect event handlers
        self._connect_event_handlers()
        
        # Build the interface
        self._build_interface()
    
    def _update_prompt_and_chain(self):
        """Update the prompt template and chain based on current system message."""
        current_message = getattr(self, 'system_message_input', None)
        if current_message:
            system_message = current_message.value.strip()
        else:
            system_message = self.default_system_message
            
        if not system_message:
            system_message = self.default_system_message
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("placeholder", "{chat_history}"),
            ("human", "{input}")
        ])
        
        # Create the chain with prompt included
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    def _create_ui_components(self):
        """Create all UI widgets for the chatbot."""
        
        # Header components
        self.title = widgets.HTML("<h2 style='text-align: center; color: #4a86e8;'>Simple Chatbot</h2>")
        self.subtitle = widgets.HTML("<p style='text-align: center; color: #666;'>Open the settings tab to set the system prompt and the inference parameters.</p>")
        
        # System message components
        self.system_message_label = widgets.HTML("<p style='margin-bottom: 5px;'><b>System Message</b> (optional):</p>")
        self.system_message_input = widgets.Textarea(
            value=self.default_system_message,
            placeholder='Define the AI assistant behavior...',
            layout={'width': '100%', 'height': '80px'}
        )
        
        # Inference parameters components
        self.inference_label = widgets.HTML("<p style='margin: 15px 0 5px 0;'><b>Inference Parameters</b>:</p>")
        
        self.temperature_label = widgets.HTML(
            "<p style='margin: 5px 0 0 0; font-size: 0.9em;'>Temperature (0 = deterministic, 1 = creative):</p>"
        )
        self.temperature_slider = widgets.FloatSlider(
            value=self.default_temperature,
            min=0,
            max=1.0,
            step=0.1,
            description='',
            layout={'width': '100%'}
        )
        
        self.max_tokens_label = widgets.HTML(
            "<p style='margin: 5px 0 0 0; font-size: 0.9em;'>Max Tokens (0 = no limit):</p>"
        )
        self.max_tokens_slider = widgets.IntSlider(
            value=self.default_max_tokens,
            min=0,
            max=4000,
            step=100,
            description='',
            layout={'width': '100%'}
        )
        
        self.update_params_button = widgets.Button(
            description='Update Parameters',
            button_style='info',
            layout={'width': '100%', 'margin': '10px 0px 5px 0px'}
        )
        
        # Chat area
        self.output_area = widgets.Output(
            layout={
                'border': '1px solid #ddd', 
                'min-height': '350px', 
                'width': '100%', 
                'overflow_y': 'auto',
                'border-radius': '8px',
                'padding': '10px',
                'margin': '10px 0',
                'background-color': '#ffffff'
            }
        )
        
        # Input area
        self.input_box = widgets.Text(
            placeholder='Type your message here and press Enter...',
            layout={'width': '80%', 'margin': '10px 0px', 'padding': '8px'}
        )
        
        self.send_button = widgets.Button(
            description='Send',
            button_style='primary',
            icon='paper-plane',
            layout={'width': '18%', 'margin': '10px 0px 10px 2%'}
        )
        
        self.clear_button = widgets.Button(
            description='Clear Chat',
            button_style='danger',
            icon='trash',
            layout={'width': '100%', 'margin': '5px 0px'}
        )
        
        # Message styles
        self.user_style = "background-color: #e6f7ff; border-radius: 15px; padding: 12px; margin: 8px 0px 8px 30%; width: 60%; box-shadow: 0 1px 2px rgba(0,0,0,0.1); word-wrap: break-word; overflow-wrap: break-word;"
        self.bot_style = "background-color: #f0f0f0; border-radius: 15px; padding: 12px; margin: 8px 30% 8px 0px; width: 60%; box-shadow: 0 1px 2px rgba(0,0,0,0.1); word-wrap: break-word; overflow-wrap: break-word;"
    
    def _connect_event_handlers(self):
        """Connect UI event handlers."""
        self.send_button.on_click(self._on_send_button_clicked)
        self.clear_button.on_click(self._on_clear_button_clicked)
        self.update_params_button.on_click(self._on_update_params_clicked)
        self.input_box.on_submit(self._on_input_submit)
    
    def _build_interface(self):
        """Build the complete interface layout."""
        # Create layout sections
        header = widgets.VBox([self.title, self.subtitle])
        system_section = widgets.VBox([self.system_message_label, self.system_message_input])
        inference_section = widgets.VBox([
            self.inference_label, 
            self.temperature_label, self.temperature_slider,
            self.max_tokens_label, self.max_tokens_slider,
            self.update_params_button
        ])
        input_area = widgets.HBox([self.input_box, self.send_button])
        
        # Create settings accordion
        settings_tab = widgets.VBox([system_section, inference_section])
        self.settings_accordion = widgets.Accordion(children=[settings_tab])
        self.settings_accordion.set_title(0, 'Settings')
        self.settings_accordion.selected_index = None  # Start collapsed
        
        # Container style
        container_style = {
            'border': '1px solid #ddd', 
            'border-radius': '10px', 
            'padding': '20px'
        }
        
        # Main interface
        self.chat_interface = widgets.VBox(
            [header, self.settings_accordion, self.output_area, input_area, self.clear_button], 
            layout={'width': '900px', 'margin': '0 auto', 'padding': '10px', **container_style}
        )
    
    def _update_llm_params(self):
        """Update LLM parameters based on slider values."""
        temp = self.temperature_slider.value
        tokens = self.max_tokens_slider.value if self.max_tokens_slider.value > 0 else None
        
        # Update the LLM with new parameters
        self.llm = ChatBedrockConverse(
            model=self.model_id,
            temperature=temp,
            max_tokens=tokens,
        )
        
        # Update the chain with the new LLM
        self._update_prompt_and_chain()
    
    def _conversation_chain(self, user_input):
        """Process user input and get AI response."""
        # Get chat history from memory
        chat_history = self.memory.load_memory_variables({}).get("chat_history", "")
        
        # Generate response using the chain
        response = self.chain.invoke({
            "chat_history": chat_history,
            "input": user_input
        })
        
        # Save to memory
        self.memory.save_context({"input": user_input}, {"output": response})
        
        return response
    
    def _on_send_button_clicked(self, b):
        """Handle send button click event."""
        user_input = self.input_box.value
        self.input_box.value = ''
        
        if not user_input.strip():
            return
        
        with self.output_area:
            # Display user message with emoticon
            display(widgets.HTML(f"<div style='{self.user_style}'><b>👤 You:</b> {user_input}</div>"))
            
            # Show "thinking" indicator
            thinking = widgets.HTML("<div style='text-align: center; color: #888;'><i>Thinking...</i></div>")
            display(thinking)
            
            try:
                # Get response from the model
                response = self._conversation_chain(user_input)
                
                # Remove thinking indicator and display response
                clear_output(wait=True)
                display(widgets.HTML(f"<div style='{self.user_style}'><b>👤 You:</b> {user_input}</div>"))
                display(widgets.HTML(f"<div style='{self.bot_style}'><b>🤖 Bot:</b> {response}</div>"))
            except Exception as e:
                # Handle errors gracefully
                clear_output(wait=True)
                display(widgets.HTML(f"<div style='{self.user_style}'><b>👤 You:</b> {user_input}</div>"))
                display(widgets.HTML(
                    f"<div style='{self.bot_style}; background-color: #ffebee;'><b>🤖 Bot:</b> "
                    f"Sorry, I encountered an error: {str(e)}</div>"
                ))
    
    def _on_clear_button_clicked(self, b):
        """Handle clear button click event."""
        with self.output_area:
            clear_output()
        self.memory.clear()
        # Display welcome message again
        with self.output_area:
            display(widgets.HTML(
                f"<div style='{self.bot_style}'><b>🤖 Bot:</b> Hello! I'm a simple chatbot powered by AWS Bedrock. "
                f"How can I help you today?</div>"
            ))
    
    def _on_update_params_clicked(self, b):
        """Handle update parameters button click event."""
        self._update_llm_params()
        self._update_prompt_and_chain()  # Update prompt and chain with new system message
        with self.output_area:
            temp_value = self.temperature_slider.value
            max_tokens = self.max_tokens_slider.value if self.max_tokens_slider.value > 0 else "No limit"
            display(widgets.HTML(
                f"<div style='text-align: center; color: #4a86e8; padding: 5px;'>"
                f"<i>Parameters updated: Temperature = {temp_value}, Max Tokens = {max_tokens} ✓</i></div>"
            ))
    
    def _on_input_submit(self, widget):
        """Handle input box submit event (Enter key)."""
        if self.input_box.value.strip():
            self._on_send_button_clicked(None)
    
    def display(self):
        """Display the chatbot interface and show welcome message."""
        import time
        time.sleep(0.5)  # Prevent race condition with frontend widget model registration
        display(self.chat_interface)
        
        # Welcome message
        with self.output_area:
            display(widgets.HTML(
                f"<div style='{self.bot_style}'><b>🤖 Bot:</b> Hello! I'm a simple chatbot powered by AWS Bedrock. "
                f"How can I help you today?</div>"
            ))


class ChatbotwithPdfSupport:
    """
    A chatbot class using AWS Bedrock and LangChain with ability to chat with documents.
    
    This class creates a user-friendly chatbot interface with:
    - System message customization
    - Temperature and max tokens controls
    - Conversation memory
    - PDF document upload and analysis
    - Clean, responsive UI
    """
    
    def __init__(self, model_id="amazon.nova-pro-v1:0", 
                 default_system_message="You are a helpful, friendly AI assistant. Be concise and clear in your responses.",
                 default_temperature=0,
                 default_max_tokens=1000):
        """Initialize the chatbot with model and default parameters."""
        
        # Store default values
        self.model_id = model_id
        self.default_system_message = default_system_message
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        
        # PDF content storage
        self.pdf_content = None
        self.pdf_filename = None
        
        # Initialize LLM
        self.llm = ChatBedrockConverse(
            model=self.model_id,
            temperature=self.default_temperature,
            max_tokens=None if default_max_tokens == 0 else default_max_tokens
        )
        
        # Initialize memory
        self.memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
        
        # Create the prompt template and chain
        self._update_prompt_and_chain()
        
        # Create UI components
        self._create_ui_components()
        
        # Connect event handlers
        self._connect_event_handlers()
        
        # Build the interface
        self._build_interface()
    
    def _update_prompt_and_chain(self):
        """Update the prompt template and chain based on current system message and PDF content."""
        current_message = getattr(self, 'system_message_input', None)
        if current_message:
            system_message = current_message.value.strip()
        else:
            system_message = self.default_system_message
            
        if not system_message:
            system_message = self.default_system_message
        
        # Create prompt template based on whether PDF content is available
        if self.pdf_content:
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", system_message),
                ("system", "The user has uploaded a PDF document. Here is the content of that document:\n\n{pdf_content}\n\nPlease refer to this document when answering questions about it."),
                ("placeholder", "{chat_history}"),
                ("human", "{input}")
            ])
        else:
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", system_message),
                ("placeholder", "{chat_history}"),
                ("human", "{input}")
            ])
        
        # Create the chain with prompt included
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    def _create_ui_components(self):
        """Create all UI widgets for the chatbot."""
        
        # Header components
        self.title = widgets.HTML("<h2 style='text-align: center; color: #4a86e8;'>Chatbot with PDF support</h2>")
        self.subtitle = widgets.HTML("<p style='text-align: center; color: #666;'>Open the settings tab and upload your PDF to chat with it.</p>")
        
        # System message components
        self.system_message_label = widgets.HTML("<p style='margin-bottom: 5px;'><b>System Message</b> (optional):</p>")
        self.system_message_input = widgets.Textarea(
            value=self.default_system_message,
            placeholder='Define the AI assistant behavior...',
            layout={'width': '100%', 'height': '80px'}
        )
        
        # PDF upload components
        self.pdf_upload_label = widgets.HTML("<p style='margin: 15px 0 5px 0;'><b>Upload PDF Document</b> (optional):</p>")
        self.pdf_upload = widgets.FileUpload(
            accept='.pdf',
            multiple=False,
            description='Select PDF',
            layout={'width': '100%'}
        )
        self.pdf_status = widgets.HTML(
            "<p style='margin: 5px 0; font-size: 0.9em; color: #888;'>No PDF uploaded</p>"
        )
        self.clear_pdf_button = widgets.Button(
            description='Clear PDF',
            button_style='warning',
            disabled=True,
            layout={'width': '100%', 'margin': '5px 0px'}
        )
        
        # Inference parameters components
        self.inference_label = widgets.HTML("<p style='margin: 15px 0 5px 0;'><b>Inference Parameters</b>:</p>")
        
        self.temperature_label = widgets.HTML(
            "<p style='margin: 5px 0 0 0; font-size: 0.9em;'>Temperature (0 = deterministic, 1 = creative):</p>"
        )
        self.temperature_slider = widgets.FloatSlider(
            value=self.default_temperature,
            min=0,
            max=1.0,
            step=0.1,
            description='',
            layout={'width': '100%'}
        )
        
        self.max_tokens_label = widgets.HTML(
            "<p style='margin: 5px 0 0 0; font-size: 0.9em;'>Max Tokens (0 = no limit):</p>"
        )
        self.max_tokens_slider = widgets.IntSlider(
            value=self.default_max_tokens,
            min=0,
            max=4000,
            step=100,
            description='',
            layout={'width': '100%'}
        )
        
        self.update_params_button = widgets.Button(
            description='Update Parameters',
            button_style='info',
            layout={'width': '100%', 'margin': '10px 0px 5px 0px'}
        )
        
        # Chat area
        self.output_area = widgets.Output(
            layout={
                'border': '1px solid #ddd', 
                'min-height': '350px', 
                'width': '100%', 
                'overflow_y': 'auto',
                'border-radius': '8px',
                'padding': '10px',
                'margin': '10px 0',
                'background-color': '#ffffff'
            }
        )
        
        # Input area
        self.input_box = widgets.Text(
            placeholder='Type your message here and press Enter...',
            layout={'width': '80%', 'margin': '10px 0px', 'padding': '8px'}
        )
        
        self.send_button = widgets.Button(
            description='Send',
            button_style='primary',
            icon='paper-plane',
            layout={'width': '18%', 'margin': '10px 0px 10px 2%'}
        )
        
        self.clear_button = widgets.Button(
            description='Clear Chat',
            button_style='danger',
            icon='trash',
            layout={'width': '100%', 'margin': '5px 0px'}
        )
        
        # Message styles
        self.user_style = "background-color: #e6f7ff; border-radius: 15px; padding: 12px; margin: 8px 0px 8px 30%; width: 60%; box-shadow: 0 1px 2px rgba(0,0,0,0.1); word-wrap: break-word; overflow-wrap: break-word;"
        self.bot_style = "background-color: #f0f0f0; border-radius: 15px; padding: 12px; margin: 8px 30% 8px 0px; width: 60%; box-shadow: 0 1px 2px rgba(0,0,0,0.1); word-wrap: break-word; overflow-wrap: break-word;"
        self.system_style = "background-color: #f9f9f9; border-radius: 8px; padding: 8px; margin: 5px 0; text-align: center; color: #666; font-style: italic;"
    
    def _connect_event_handlers(self):
        """Connect UI event handlers."""
        self.send_button.on_click(self._on_send_button_clicked)
        self.clear_button.on_click(self._on_clear_button_clicked)
        self.update_params_button.on_click(self._on_update_params_clicked)
        self.input_box.on_submit(self._on_input_submit)
        self.pdf_upload.observe(self._on_pdf_upload, names='value')
        self.clear_pdf_button.on_click(self._on_clear_pdf_clicked)
    
    def _build_interface(self):
        """Build the complete interface layout."""
        # Create layout sections
        header = widgets.VBox([self.title, self.subtitle])
        system_section = widgets.VBox([self.system_message_label, self.system_message_input])
        pdf_section = widgets.VBox([
            self.pdf_upload_label, 
            self.pdf_upload, 
            self.pdf_status,
            self.clear_pdf_button
        ])
        inference_section = widgets.VBox([
            self.inference_label, 
            self.temperature_label, self.temperature_slider,
            self.max_tokens_label, self.max_tokens_slider,
            self.update_params_button
        ])
        input_area = widgets.HBox([self.input_box, self.send_button])
        
        # Create settings accordion
        settings_tab = widgets.VBox([system_section, pdf_section, inference_section])
        self.settings_accordion = widgets.Accordion(children=[settings_tab])
        self.settings_accordion.set_title(0, 'Settings')
        self.settings_accordion.selected_index = None  # Start collapsed
        
        # Container style
        container_style = {
            'border': '1px solid #ddd', 
            'border-radius': '10px', 
            'padding': '20px'
        }
        
        # Main interface
        self.chat_interface = widgets.VBox(
            [header, self.settings_accordion, self.output_area, input_area, self.clear_button], 
            layout={'width': '900px', 'margin': '0 auto', 'padding': '10px', **container_style}
        )
    
    def _update_llm_params(self):
        """Update LLM parameters based on slider values."""
        temp = self.temperature_slider.value
        tokens = self.max_tokens_slider.value if self.max_tokens_slider.value > 0 else None
        
        # Update the LLM with new parameters
        self.llm = ChatBedrockConverse(
            model=self.model_id,
            temperature=temp,
            max_tokens=tokens,
        )
        
        # Update the chain with the new LLM
        self._update_prompt_and_chain()
    
    def _extract_text_from_pdf(self, pdf_data):
        """Extract text from PDF data (limited to first 3 pages)."""
        try:
            # Create a PDF reader object
            pdf_file = io.BytesIO(pdf_data)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # Extract text from each page (up to 3 pages only)
            text = ""
            max_pages = min(3, len(pdf_reader.pages))
            for page_num in range(max_pages):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n\n"
            
            # Add note if document was truncated
            if len(pdf_reader.pages) > 3:
                text += "\n[Note: This document has been truncated. Only the first 3 pages are included.]"
            
            return text
        except Exception as e:
            return f"Error extracting text from PDF: {str(e)}"
    
    def _on_pdf_upload(self, change):
        """Handle PDF upload event."""
        if change['new'] and len(change['new']) > 0:
            # Get the uploaded file
            uploaded_file = self.pdf_upload.value[0]
            filename = uploaded_file['name']
            content = uploaded_file['content']
            
            # Extract text from the PDF
            extracted_text = self._extract_text_from_pdf(content)
            
            # Store the PDF content and filename
            self.pdf_content = extracted_text
            self.pdf_filename = filename
            
            # Update the PDF status
            self.pdf_status.value = f"<p style='margin: 5px 0; font-size: 0.9em; color: #4a86e8;'><b>✓</b> PDF loaded: {filename} ({len(extracted_text)} characters)</p>"
            
            # Enable the clear PDF button
            self.clear_pdf_button.disabled = False
            
            # Update the prompt template to include PDF content
            self._update_prompt_and_chain()
            
            # Show notification in chat
            with self.output_area:
                display(widgets.HTML(
                    f"<div style='{self.system_style}'>PDF document loaded: {filename}</div>"
                ))
    
    def _on_clear_pdf_clicked(self, b):
        """Handle clear PDF button click event."""
        # Clear PDF content and filename
        self.pdf_content = None
        self.pdf_filename = None
        
        # Update the PDF status
        self.pdf_status.value = "<p style='margin: 5px 0; font-size: 0.9em; color: #888;'>No PDF uploaded</p>"
        
        # Disable the clear PDF button
        self.clear_pdf_button.disabled = True
        
        # Clear the file upload widget
        self.pdf_upload.value = ()
        
        # Update the prompt template to remove PDF content
        self._update_prompt_and_chain()
        
        # Show notification in chat
        with self.output_area:
            display(widgets.HTML(
                f"<div style='{self.system_style}'>PDF document removed</div>"
            ))
    
    def _conversation_chain(self, user_input):
        """Process user input and get AI response."""
        # Get chat history from memory
        chat_history = self.memory.load_memory_variables({}).get("chat_history", "")
        
        # Prepare input variables
        input_vars = {
            "chat_history": chat_history,
            "input": user_input
        }
        
        # Add PDF content if available
        if self.pdf_content:
            input_vars["pdf_content"] = self.pdf_content
        
        # Generate response using the chain
        response = self.chain.invoke(input_vars)
        
        # Save to memory
        self.memory.save_context({"input": user_input}, {"output": response})
        
        return response
    
    def _on_send_button_clicked(self, b):
        """Handle send button click event."""
        user_input = self.input_box.value
        self.input_box.value = ''
        
        if not user_input.strip():
            return
        
        with self.output_area:
            # Display user message with emoticon
            display(widgets.HTML(f"<div style='{self.user_style}'><b>👤 You:</b> {user_input}</div>"))
            
            # Show "thinking" indicator
            thinking = widgets.HTML("<div style='text-align: center; color: #888;'><i>Responding...</i></div>")
            display(thinking)
            
            try:
                # Get response from the model
                response = self._conversation_chain(user_input)
                
                # Remove thinking indicator and display response
                clear_output(wait=True)
                display(widgets.HTML(f"<div style='{self.user_style}'><b>👤 You:</b> {user_input}</div>"))
                display(widgets.HTML(f"<div style='{self.bot_style}'><b>🤖 Bot:</b> {response}</div>"))
            except Exception as e:
                # Handle errors gracefully
                clear_output(wait=True)
                display(widgets.HTML(f"<div style='{self.user_style}'><b>👤 You:</b> {user_input}</div>"))
                display(widgets.HTML(
                    f"<div style='{self.bot_style}; background-color: #ffebee;'><b>🤖 Bot:</b> "
                    f"Sorry, I encountered an error: {str(e)}</div>"
                ))
    
    def _on_clear_button_clicked(self, b):
        """Handle clear button click event."""
        with self.output_area:
            clear_output()
        self.memory.clear()
        # Display welcome message again
        with self.output_area:
            welcome_message = "Hello! I'm a chatbot capable of chatting with your document."
            if self.pdf_content:
                welcome_message += f" I have a PDF document loaded ({self.pdf_filename}). You can ask me questions about it!"
            else:
                welcome_message += " How can I help you today?"
            
            display(widgets.HTML(
                f"<div style='{self.bot_style}'><b>🤖 Bot:</b> {welcome_message}</div>"
            ))
    
    def _on_update_params_clicked(self, b):
        """Handle update parameters button click event."""
        self._update_llm_params()
        self._update_prompt_and_chain()  # Update prompt and chain with new system message
        with self.output_area:
            temp_value = self.temperature_slider.value
            max_tokens = self.max_tokens_slider.value if self.max_tokens_slider.value > 0 else "No limit"
            display(widgets.HTML(
                f"<div style='{self.system_style}'>"
                f"Parameters updated: Temperature = {temp_value}, Max Tokens = {max_tokens} ✓</div>"
            ))
    
    def _on_input_submit(self, widget):
        """Handle input box submit event (Enter key)."""
        if self.input_box.value.strip():
            self._on_send_button_clicked(None)
    
    def display(self):
        """Display the chatbot interface and show welcome message."""
        import time
        time.sleep(0.5)  # Prevent race condition with frontend widget model registration
        display(self.chat_interface)
        
        # Welcome message
        with self.output_area:
            welcome_message = "Hello! I'm a chatbot capable of chatting with your document."
            display(widgets.HTML(
                f"<div style='{self.bot_style}'><b>🤖 Bot:</b> {welcome_message}</div>"
            ))