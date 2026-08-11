from typing import Any, Dict, List, Optional, Sequence, Union

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    get_buffer_string,
)
from pydantic import BaseModel, Field, root_validator, ConfigDict
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate


class ConversationBufferMemory(BaseModel):
    """Memory that stores chat message history in a simple buffer.
    
    This memory utilizes BaseChatMessageHistory for storing the conversation.
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    chat_memory: BaseChatMessageHistory = Field(default_factory=lambda: InMemoryChatMessageHistory())
    """The chat message history to use to store and retrieve messages."""
    
    return_messages: bool = False
    """Whether to return the chat messages directly or convert them to a string."""
    
    output_key: Optional[str] = None
    """If set, the output of the model will be saved to this key in memory."""
    
    input_key: Optional[str] = None
    """If set, the input to the model will be saved to this key in memory."""
    
    human_prefix: str = "Human"
    """Prefix to use for human messages."""
    
    ai_prefix: str = "AI"
    """Prefix to use for AI messages."""
    
    memory_key: str = "history"
    """Key to use for the memory in the chain input."""

    @property
    def memory_variables(self) -> List[str]:
        """Return the memory variables."""
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load the memory variables from the chat history."""
        if self.return_messages:
            return {self.memory_key: self.chat_memory.messages}
        else:
            return {self.memory_key: get_buffer_string(
                self.chat_memory.messages,
                human_prefix=self.human_prefix,
                ai_prefix=self.ai_prefix,
            )}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """Save the context of this conversation to the chat history."""
        # Save the input and output to the memory
        input_str = self._get_input_value(inputs)
        output_str = self._get_output_value(outputs)
        
        # Add the messages to the chat history
        self.chat_memory.add_messages([
            HumanMessage(content=input_str),
            AIMessage(content=output_str)
        ])

    def _get_input_value(self, inputs: Dict[str, Any]) -> str:
        """Get the input value from the inputs dictionary."""
        if self.input_key is None:
            # If input_key is None, assume single input value
            if len(inputs) != 1:
                raise ValueError(f"input_key not specified and inputs has multiple keys: {inputs.keys()}")
            return str(next(iter(inputs.values())))
        else:
            # If input_key is specified, use that
            if self.input_key not in inputs:
                raise ValueError(f"input_key {self.input_key} not found in inputs: {inputs.keys()}")
            return str(inputs[self.input_key])

    def _get_output_value(self, outputs: Dict[str, Any]) -> str:
        """Get the output value from the outputs dictionary."""
        if self.output_key is None:
            # If output_key is None, assume single output value
            if len(outputs) != 1:
                raise ValueError(f"output_key not specified and outputs has multiple keys: {outputs.keys()}")
            return str(next(iter(outputs.values())))
        else:
            # If output_key is specified, use that
            if self.output_key not in outputs:
                raise ValueError(f"output_key {self.output_key} not found in outputs: {outputs.keys()}")
            return str(outputs[self.output_key])

    def clear(self) -> None:
        """Clear the memory contents."""
        self.chat_memory.clear()


class ConversationBufferWindowMemory(BaseModel):
    """Memory that keeps a sliding window of chat message history.
    
    This memory only keeps the last K interactions in memory, where an interaction
    consists of a human message and the corresponding AI message.
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    chat_memory: BaseChatMessageHistory = Field(default_factory=lambda: InMemoryChatMessageHistory())
    """The chat message history to use to store and retrieve messages."""
    
    k: int = 5
    """Number of conversation interactions to keep in memory."""
    
    return_messages: bool = False
    """Whether to return the chat messages directly or convert them to a string."""
    
    output_key: Optional[str] = None
    """If set, the output of the model will be saved to this key in memory."""
    
    input_key: Optional[str] = None
    """If set, the input to the model will be saved to this key in memory."""
    
    human_prefix: str = "Human"
    """Prefix to use for human messages."""
    
    ai_prefix: str = "AI"
    """Prefix to use for AI messages."""
    
    memory_key: str = "history"
    """Key to use for the memory in the chain input."""

    @property
    def memory_variables(self) -> List[str]:
        """Return the memory variables."""
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load the memory variables from the chat history."""
        # Get the messages from the chat history
        messages = self.chat_memory.messages
        
        # Apply the window to get the most recent k interactions (2 messages per interaction)
        window_messages = messages[-2 * self.k:] if messages else []
        
        if self.return_messages:
            return {self.memory_key: window_messages}
        else:
            return {self.memory_key: get_buffer_string(
                window_messages,
                human_prefix=self.human_prefix,
                ai_prefix=self.ai_prefix,
            )}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """Save the context of this conversation to the chat history."""
        # Save the input and output to the memory
        input_str = self._get_input_value(inputs)
        output_str = self._get_output_value(outputs)
        
        # Add the messages to the chat history
        self.chat_memory.add_messages([
            HumanMessage(content=input_str),
            AIMessage(content=output_str)
        ])

    def _get_input_value(self, inputs: Dict[str, Any]) -> str:
        """Get the input value from the inputs dictionary."""
        if self.input_key is None:
            # If input_key is None, assume single input value
            if len(inputs) != 1:
                raise ValueError(f"input_key not specified and inputs has multiple keys: {inputs.keys()}")
            return str(next(iter(inputs.values())))
        else:
            # If input_key is specified, use that
            if self.input_key not in inputs:
                raise ValueError(f"input_key {self.input_key} not found in inputs: {inputs.keys()}")
            return str(inputs[self.input_key])

    def _get_output_value(self, outputs: Dict[str, Any]) -> str:
        """Get the output value from the outputs dictionary."""
        if self.output_key is None:
            # If output_key is None, assume single output value
            if len(outputs) != 1:
                raise ValueError(f"output_key not specified and outputs has multiple keys: {outputs.keys()}")
            return str(next(iter(outputs.values())))
        else:
            # If output_key is specified, use that
            if self.output_key not in outputs:
                raise ValueError(f"output_key {self.output_key} not found in outputs: {outputs.keys()}")
            return str(outputs[self.output_key])

    def clear(self) -> None:
        """Clear the memory contents."""
        self.chat_memory.clear()


class ConversationSummaryMemory(BaseModel):
    """Memory that keeps a summary of the conversation and uses an LLM to generate it.
    
    This memory summarizes the conversation as it happens and uses the summary
    as context for the next conversation turn.
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    chat_memory: BaseChatMessageHistory = Field(default_factory=lambda: InMemoryChatMessageHistory())
    """The chat message history to use to store and retrieve messages."""
    
    llm: BaseLanguageModel
    """The language model to use for summarization."""
    
    summary: str = ""
    """The current summary of the conversation."""
    
    return_messages: bool = False
    """Whether to return the chat messages directly or convert them to a string."""
    
    output_key: Optional[str] = None
    """If set, the output of the model will be saved to this key in memory."""
    
    input_key: Optional[str] = None
    """If set, the input to the model will be saved to this key in memory."""
    
    human_prefix: str = "Human"
    """Prefix to use for human messages."""
    
    ai_prefix: str = "AI"
    """Prefix to use for AI messages."""
    
    memory_key: str = "history"
    """Key to use for the memory in the chain input."""
    
    prompt: PromptTemplate = Field(default_factory=lambda: PromptTemplate.from_template(
        """Progressively summarize the lines of conversation provided, adding onto the previous summary returning a new summary.

Previous summary:
{summary}

New lines of conversation:
{new_lines}

New summary:"""
    ))
    """The prompt to use for summarization."""

    @property
    def memory_variables(self) -> List[str]:
        """Return the memory variables."""
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load the memory variables from the chat history."""
        if self.return_messages:
            # If returning messages, create a system message with the summary
            if self.summary:
                return {self.memory_key: [SystemMessage(content=self.summary)]}
            else:
                return {self.memory_key: []}
        else:
            # If returning a string, just return the summary
            return {self.memory_key: self.summary}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """Save the context of this conversation to the chat history."""
        # Save the input and output to the memory
        input_str = self._get_input_value(inputs)
        output_str = self._get_output_value(outputs)
        
        # Format the new lines of conversation
        new_lines = f"{self.human_prefix}: {input_str}\n{self.ai_prefix}: {output_str}"
        
        # Update the summary
        self.summary = self._update_summary(new_lines)
        
        # Add the messages to the chat history
        self.chat_memory.add_messages([
            HumanMessage(content=input_str),
            AIMessage(content=output_str)
        ])

    def _update_summary(self, new_lines: str) -> str:
        """Update the summary based on the new lines of conversation."""
        if not self.summary:
            # If there's no summary yet, just use the new lines as the summary
            chain_input = {"summary": "", "new_lines": new_lines}
        else:
            # Otherwise, update the summary with the new lines
            chain_input = {"summary": self.summary, "new_lines": new_lines}
        
        # Generate the new summary using the LLM
        return self.llm.predict(self.prompt.format(**chain_input))

    def _get_input_value(self, inputs: Dict[str, Any]) -> str:
        """Get the input value from the inputs dictionary."""
        if self.input_key is None:
            # If input_key is None, assume single input value
            if len(inputs) != 1:
                raise ValueError(f"input_key not specified and inputs has multiple keys: {inputs.keys()}")
            return str(next(iter(inputs.values())))
        else:
            # If input_key is specified, use that
            if self.input_key not in inputs:
                raise ValueError(f"input_key {self.input_key} not found in inputs: {inputs.keys()}")
            return str(inputs[self.input_key])

    def _get_output_value(self, outputs: Dict[str, Any]) -> str:
        """Get the output value from the outputs dictionary."""
        if self.output_key is None:
            # If output_key is None, assume single output value
            if len(outputs) != 1:
                raise ValueError(f"output_key not specified and outputs has multiple keys: {outputs.keys()}")
            return str(next(iter(outputs.values())))
        else:
            # If output_key is specified, use that
            if self.output_key not in outputs:
                raise ValueError(f"output_key {self.output_key} not found in outputs: {outputs.keys()}")
            return str(outputs[self.output_key])

    def clear(self) -> None:
        """Clear the memory contents."""
        self.chat_memory.clear()
        self.summary = ""