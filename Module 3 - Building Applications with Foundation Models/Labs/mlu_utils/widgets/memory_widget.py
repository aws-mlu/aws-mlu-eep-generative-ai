import html

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
from strands import Agent
from strands.agent.conversation_manager import (
    NullConversationManager,
    SlidingWindowConversationManager,
    SummarizingConversationManager,
)
import warnings

warnings.filterwarnings("ignore")

SYSTEM_PROMPT = "You are a helpful AI assistant."

MEMORY_DEMO_CSS = """
    <style>
        .memory-demo-container { max-width: 1000px; margin: 0 auto; }
        .section-header { color: #2c3e50; font-weight: 600; margin-top: 20px; 
                         margin-bottom: 12px; padding-bottom: 8px; border-bottom: 2px solid #ebf5fb; }
        .widget-button button { background-color: #3498db !important; color: white !important; 
                               border: none !important; border-radius: 4px !important; }
        .widget-button button:hover:enabled { background-color: #2980b9 !important; }
        .widget-button button:disabled { background-color: #bdc3c7 !important; opacity: 0.7 !important; }
        .conversation-area { background-color: #f8f9fa; border-radius: 6px; padding: 15px; 
                            border: 1px solid #e0e0e0; font-family: sans-serif; }
        .memory-area { background-color: #f0f7fb; border-radius: 6px; padding: 15px; 
                      border: 1px solid #d1e3f0; font-family: monospace; }
        .user-message { background-color: #e8f4fd; border-radius: 18px; padding: 10px 15px; 
                       margin: 5px 0; display: inline-block; max-width: 80%; }
        .ai-message { background-color: #f0f0f0; border-radius: 18px; padding: 10px 15px; 
                     margin: 5px 0 15px auto; display: inline-block; max-width: 80%; }
        .message-container { margin: 10px 0; }
        .user-container { text-align: left; }
        .ai-container { text-align: right; }
        .memory-content { white-space: pre-wrap; }
        .memory-header { color: #3498db; font-weight: bold; margin-top: 10px; }
        .memory-item { margin: 5px 0; padding: 5px; border-bottom: 1px solid #e0e0e0; }
        .parameter-container { margin-top: 10px; padding: 10px; background-color: #f8f9fa; border-radius: 6px; }
        .parameter-label { font-weight: bold; margin-bottom: 5px; }
    </style>
    """


def _message_text(message):
    """Return the text of a Strands message (a dict with a role and a list of content blocks)."""
    return "".join(block.get("text", "") for block in message.get("content", []))


def _format_messages(messages):
    """Format the messages that the agent keeps in memory as 'Human: ...' and 'AI: ...' lines."""
    lines = []
    for message in messages:
        speaker = "Human" if message["role"] == "user" else "AI"
        lines.append(f"{speaker}: {html.escape(_message_text(message))}")
    return "\n".join(lines) + "\n"


def _conversation_html(user_msg, ai_msg):
    return f"""
                <div class="message-container user-container">
                    <div class="user-message">👤 <strong>User:</strong> {html.escape(user_msg)}</div>
                </div>
                <div class="message-container ai-container">
                    <div class="ai-message">🤖 <strong>AI:</strong> {html.escape(ai_msg)}</div>
                </div>
                """


def create_memory_demo_ui(model):
    """Chat demo that shows the messages that the Strands agent keeps in its memory (`agent.messages`).

    Args:
        model: The Strands model (for example a BedrockModel) used by the agent.
    """
    # Apply styling for a cleaner UI
    display(HTML(MEMORY_DEMO_CSS))

    # The agent keeps the conversation history in agent.messages.
    # NullConversationManager does not remove or change any of the messages.
    agent = Agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        conversation_manager=NullConversationManager(),
        callback_handler=None,
    )

    # Create widgets
    prompt_input = widgets.Text(
        value='',
        placeholder='Type your prompt here and press Enter or click Submit...',
        description='',
        layout=widgets.Layout(width='100%')
    )

    submit_button = widgets.Button(
        description='Submit',
        icon='paper-plane',
        button_style='primary',
        layout=widgets.Layout(width='150px')
    )

    conversation_output = widgets.Output(layout=widgets.Layout(
        width='100%',
        min_height='300px',
        max_height='500px',
        overflow='auto'
    ))

    memory_output = widgets.Output(layout=widgets.Layout(
        width='100%',
        min_height='300px',
        max_height='500px',
        overflow='auto'
    ))

    # History tracking
    conversation_history = []

    # Function to format the conversation nicely
    def format_conversation():
        with conversation_output:
            clear_output(wait=True)
            for user_msg, ai_msg in conversation_history:
                display(HTML(_conversation_html(user_msg, ai_msg)))

    # Function to update the memory display
    def update_memory_display():
        with memory_output:
            clear_output(wait=True)
            display(HTML(f"""
            <div class="memory-content memory-item">{_format_messages(agent.messages)}</div>
            """))

    # Handle submission
    def on_submit(b=None):
        prompt = prompt_input.value
        if not prompt.strip():
            return

        # Clear the input field
        prompt_input.value = ''

        # Get response from the agent. The agent adds the prompt and the response to its messages.
        response = str(agent(prompt))

        # Add to history
        conversation_history.append((prompt, response))

        # Update displays
        format_conversation()
        update_memory_display()

    # Connect the button click and Enter key to the handler
    submit_button.on_click(on_submit)
    prompt_input.on_submit(on_submit)

    # Create layout
    input_box = widgets.HBox([
        prompt_input,
        widgets.Box([submit_button], layout=widgets.Layout(display='flex', align_items='center'))
    ], layout=widgets.Layout(width='100%', margin='15px 0'))

    # Initialize displays
    format_conversation()
    update_memory_display()

    # Assemble the UI
    return widgets.VBox([
        widgets.HTML("<h2 class='section-header'>Memory Widget</h2>"),
        widgets.HTML("""
        <p style="color:#555; margin-bottom:20px;">
            This demo shows how memory works in an agent. Type prompts to interact with the AI 
            and observe how the agent keeps track of the conversation history in its messages.
        </p>
        """),
        widgets.HBox([
            widgets.VBox([
                widgets.HTML("<h3 class='section-header'>Conversation</h3>"),
                widgets.Box([conversation_output], layout=widgets.Layout(
                    border='1px solid #e0e0e0',
                    border_radius='6px',
                    padding='0',
                    width='100%',
                    class_='conversation-area'
                ))
            ], layout=widgets.Layout(width='50%', padding='0 10px 0 0')),

            widgets.VBox([
                widgets.HTML("<h3 class='section-header'>🧠 MEMORY CONTENT (agent.messages)</h3>"),
                widgets.Box([memory_output], layout=widgets.Layout(
                    border='1px solid #d1e3f0',
                    border_radius='6px',
                    padding='0',
                    width='100%',
                    class_='memory-area'
                ))
            ], layout=widgets.Layout(width='50%', padding='0 0 0 10px'))
        ], layout=widgets.Layout(width='100%')),

        widgets.HTML("<h3 class='section-header'>New Prompt</h3>"),
        input_box
    ], layout=widgets.Layout(
        padding='25px',
        border='1px solid #e0e0e0',
        border_radius='8px',
        width='100%',
        max_width='1000px',
        margin='0 auto',
        class_='memory-demo-container'
    ))


def create_configurable_memory_demo_ui(model):
    """Chat demo that lets you pick the Strands conversation manager of the agent and see its memory.

    Args:
        model: The Strands model (for example a BedrockModel) used by the agent.
    """
    # Apply styling for a cleaner UI
    display(HTML(MEMORY_DEMO_CSS))

    # Conversation manager selection
    manager_dropdown = widgets.Dropdown(
        options=[
            'NullConversationManager',
            'SlidingWindowConversationManager',
            'SummarizingConversationManager'
        ],
        value='NullConversationManager',
        description='Manager:',
        layout=widgets.Layout(width='400px')
    )

    # Parameter widgets for the different conversation managers
    window_size_param = widgets.IntText(
        value=6,
        description='window_size:',
        disabled=True,
        layout=widgets.Layout(width='200px')
    )
    summary_ratio_param = widgets.FloatText(
        value=0.5,
        description='summary_ratio:',
        disabled=True,
        layout=widgets.Layout(width='200px')
    )
    preserve_recent_param = widgets.IntText(
        value=2,
        description='preserve_recent:',
        disabled=True,
        layout=widgets.Layout(width='200px')
    )
    summarize_after_param = widgets.IntText(
        value=6,
        description='summarize_after:',
        disabled=True,
        layout=widgets.Layout(width='200px')
    )

    # Apply button to update the conversation manager
    apply_button = widgets.Button(
        description='Apply Memory Settings',
        button_style='primary',
        icon='check',
        layout=widgets.Layout(width='200px')
    )

    # Container for parameter widgets
    param_container = widgets.VBox([
        widgets.HTML("<p class='parameter-label'>Memory Parameters:</p>"),
        widgets.HBox([window_size_param, summary_ratio_param, preserve_recent_param, summarize_after_param]),
        widgets.HTML(
            "<p style='color:#555; font-size: 0.85em; margin: 8px 0 0 0;'>"
            "<b>window_size</b>: maximum number of messages kept (2 messages per exchange). "
            "<b>summary_ratio</b> and <b>preserve_recent</b>: the share of the messages that is summarized and the "
            "number of most recent messages that are always kept as they are. "
            "<b>summarize_after</b>: in a real application, the summary is created when the context window of the model "
            "is about to overflow. In this demo, we trigger it when the history has more than this number of messages.</p>"
        ),
    ], layout=widgets.Layout(
        margin='10px 0',
        padding='10px',
        border='1px solid #e0e0e0',
        border_radius='6px',
        class_='parameter-container'
    ))

    def create_conversation_manager():
        """Create the conversation manager that is selected in the widget."""
        if manager_dropdown.value == 'SlidingWindowConversationManager':
            return SlidingWindowConversationManager(window_size=window_size_param.value)
        if manager_dropdown.value == 'SummarizingConversationManager':
            return SummarizingConversationManager(
                summary_ratio=summary_ratio_param.value,
                preserve_recent_messages=preserve_recent_param.value,
            )
        return NullConversationManager()

    def create_agent():
        return Agent(
            model=model,
            system_prompt=SYSTEM_PROMPT,
            conversation_manager=create_conversation_manager(),
            callback_handler=None,
        )

    # The agent is replaced when the memory settings are applied
    state = {"agent": create_agent(), "manager": manager_dropdown.value}

    # Create widgets for conversation
    prompt_input = widgets.Text(
        value='',
        placeholder='Type your prompt here and press Enter or click Submit...',
        description='',
        layout=widgets.Layout(width='100%')
    )

    submit_button = widgets.Button(
        description='Submit',
        icon='paper-plane',
        button_style='primary',
        layout=widgets.Layout(width='150px')
    )

    reset_button = widgets.Button(
        description='Reset Conversation',
        icon='refresh',
        button_style='warning',
        layout=widgets.Layout(width='180px')
    )

    conversation_output = widgets.Output(layout=widgets.Layout(
        width='100%',
        min_height='300px',
        max_height='500px',
        overflow='auto'
    ))

    memory_output = widgets.Output(layout=widgets.Layout(
        width='100%',
        min_height='300px',
        max_height='500px',
        overflow='auto'
    ))

    # History tracking: everything the user saw, independently of what the agent keeps in memory
    conversation_history = []

    # Function to enable the parameters of the selected conversation manager
    def update_parameter_visibility(manager_name):
        window_size_param.disabled = manager_name != 'SlidingWindowConversationManager'
        is_summarizing = manager_name == 'SummarizingConversationManager'
        summary_ratio_param.disabled = not is_summarizing
        preserve_recent_param.disabled = not is_summarizing
        summarize_after_param.disabled = not is_summarizing

    # Update parameters when the conversation manager changes
    def on_manager_change(change):
        update_parameter_visibility(change.new)

    manager_dropdown.observe(on_manager_change, names='value')

    # Function to format the conversation nicely
    def format_conversation():
        with conversation_output:
            clear_output(wait=True)
            for user_msg, ai_msg in conversation_history:
                display(HTML(_conversation_html(user_msg, ai_msg)))

    # Function to update the memory display: the messages that the model receives in the next turn
    def update_memory_display():
        with memory_output:
            clear_output(wait=True)

            manager_name = state["manager"]
            messages = state["agent"].messages
            if manager_name == 'SlidingWindowConversationManager':
                title = f"Sliding window (window_size={state['agent'].conversation_manager.window_size})"
            elif manager_name == 'SummarizingConversationManager':
                title = "Summarizing (older messages are replaced by a summary)"
            else:
                title = "Null (all the messages are kept)"

            display(HTML(f"""
            <div class="memory-header">{title}: {len(messages)} messages</div>
            <div class="memory-content memory-item">{_format_messages(messages)}</div>
            """))

    # Handle conversation manager update
    def on_apply_settings(b):
        # Create a new agent with the selected conversation manager, which starts with an empty memory
        state["agent"] = create_agent()
        state["manager"] = manager_dropdown.value

        # Reset conversation history
        conversation_history.clear()

        # Update displays
        format_conversation()
        update_memory_display()

        # Show confirmation
        with memory_output:
            clear_output(wait=True)
            display(HTML(f"""
            <div style="color: green; font-weight: bold; padding: 10px; background-color: #e8f5e9; border-radius: 4px;">
                Memory settings applied successfully! Conversation manager: {manager_dropdown.value}
            </div>
            """))

    # Handle submission
    def on_submit(b=None):
        prompt = prompt_input.value
        if not prompt.strip():
            return

        # Clear the input field
        prompt_input.value = ''

        try:
            agent = state["agent"]

            # Get response from the agent. The conversation manager is applied after the agent responds.
            response = str(agent(prompt))

            # Demo only: the summarizing manager summarizes the oldest messages when the context window overflows.
            # We trigger it manually, so that the summary can be observed in a short conversation.
            if state["manager"] == 'SummarizingConversationManager' and len(agent.messages) > summarize_after_param.value:
                agent.conversation_manager.reduce_context(agent)

            # Add to history
            conversation_history.append((prompt, response))

            # Update displays
            format_conversation()
            update_memory_display()
        except Exception as e:
            with conversation_output:
                clear_output(wait=True)
                display(HTML(f"""
                <div style="color: red; font-weight: bold; padding: 10px; background-color: #ffebee; border-radius: 4px;">
                    Error: {html.escape(str(e))}
                </div>
                """))

    # Handle reset
    def on_reset(b):
        # Reset conversation history and the memory of the agent
        conversation_history.clear()
        state["agent"].messages.clear()

        # Update displays
        format_conversation()
        update_memory_display()

    # Connect the button clicks and Enter key to the handlers
    submit_button.on_click(on_submit)
    prompt_input.on_submit(on_submit)
    apply_button.on_click(on_apply_settings)
    reset_button.on_click(on_reset)

    # Create layout
    memory_config_section = widgets.VBox([
        widgets.HTML("<h3 class='section-header'>Memory Configuration</h3>"),
        widgets.HBox([manager_dropdown, apply_button], layout=widgets.Layout(align_items='center')),
        param_container
    ])

    input_box = widgets.HBox([
        prompt_input,
        widgets.Box([submit_button], layout=widgets.Layout(display='flex', align_items='center'))
    ], layout=widgets.Layout(width='100%', margin='15px 0'))

    # Initialize displays
    update_parameter_visibility(manager_dropdown.value)
    format_conversation()
    update_memory_display()

    # Assemble the UI
    return widgets.VBox([
        widgets.HTML("<h2 class='section-header'>Configurable Memory Widget</h2>"),
        widgets.HTML("""
        <p style="color:#555; margin-bottom:20px;">
            This demo allows you to select and configure the conversation manager of a Strands agent. Choose a conversation manager,
            set its parameters, and click "Apply Memory Settings" to update the configuration.
        </p>
        """),
        memory_config_section,
        widgets.HBox([
            widgets.VBox([
                widgets.HTML("<h3 class='section-header'>Conversation</h3>"),
                widgets.Box([conversation_output], layout=widgets.Layout(
                    border='1px solid #e0e0e0',
                    border_radius='6px',
                    padding='0',
                    width='100%',
                    class_='conversation-area'
                )),
                widgets.HBox([reset_button], layout=widgets.Layout(justify_content='flex-end', margin='10px 0'))
            ], layout=widgets.Layout(width='50%', padding='0 10px 0 0')),

            widgets.VBox([
                widgets.HTML("<h3 class='section-header'>🧠 MEMORY CONTENT (agent.messages)</h3>"),
                widgets.Box([memory_output], layout=widgets.Layout(
                    border='1px solid #d1e3f0',
                    border_radius='6px',
                    padding='0',
                    width='100%',
                    class_='memory-area'
                ))
            ], layout=widgets.Layout(width='50%', padding='0 0 0 10px'))
        ], layout=widgets.Layout(width='100%')),

        widgets.HTML("<h3 class='section-header'>New Prompt</h3>"),
        input_box
    ], layout=widgets.Layout(
        padding='25px',
        border='1px solid #e0e0e0',
        border_radius='8px',
        width='100%',
        max_width='1000px',
        margin='0 auto',
        class_='memory-demo-container'
    ))
