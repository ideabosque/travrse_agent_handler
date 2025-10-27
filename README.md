# Travrse Agent Handler

A Python module for integrating with Travrse AI to support development planning and AI-powered workflows.

## Overview

The Travrse Agent Handler provides a clean, event-driven interface for interacting with Travrse AI's API. It follows the same architectural pattern as the Ollama Agent Handler, making it easy to swap between different AI providers while maintaining consistent application logic.

## Features

- **Streaming & Non-Streaming Support**: Handle both real-time streaming responses and traditional request/response patterns
- **Tool Integration**: Support for external API tools and custom function calls
- **Conversation Management**: Automatic handling of conversation history and context
- **Performance Monitoring**: Built-in timeline logging and performance tracking
- **Error Handling**: Robust retry logic and error recovery
- **Type Safety**: Full type hints for better IDE support and code quality

## Installation

```bash
pip install -e .
```

### Dependencies

- `SilvaEngine-Utility`: Core utility functions
- `AI-Agent-Handler`: Base agent handler interface
- `requests`: HTTP client for API calls
- `pendulum`: DateTime handling

## Quick Start

### Basic Usage (Non-Streaming)

```python
import logging
from travrse_agent_handler import TravrseEventHandler

logger = logging.getLogger(__name__)

agent_config = {
    "instructions": "You are a helpful AI assistant for development planning.",
    "tool_call_role": "tool",
    "configuration": {
        "api_key": "your_travrse_api_key",
        "model": "gpt-4o",
        "response_format": "text",
    }
}

handler = TravrseEventHandler(
    logger=logger,
    agent=agent_config,
    enable_timeline_log=True
)

input_messages = [
    {
        "role": "user",
        "content": "Create a development plan for a REST API."
    }
]

run_id = handler.ask_model(input_messages=input_messages)
print(handler.final_output)
```

### Streaming Usage

**Important**: Streaming is enabled by passing a `queue` parameter to `ask_model()`, regardless of the `stream_response` configuration setting.

```python
from queue import Queue
from threading import Event

queue = Queue()
stream_event = Event()

# Passing queue=queue enables streaming
handler.ask_model(
    input_messages=input_messages,
    queue=queue,
    stream_event=stream_event
)

# Wait for streaming to complete
stream_event.wait()
print(handler.final_output)
```

**Streaming vs Non-Streaming**:
- **With queue**: `ask_model(..., queue=queue, stream_event=event)` → Streams JSON chunks
- **Without queue**: `ask_model(...)` → Returns complete response

### With External Tools

```python
agent_config = {
    "instructions": "",  # System prompt
    "tool_call_role": "tool",
    "configuration": {
        "api_key": "your_api_key",
        "model": "gpt-4o",
        "record_name": "Standalone Execution",
        "record_type": "standalone",
        "record_metadata": {"message": "Sample message"},
        "flow_name": "Onboarding Flow",
        "flow_description": "Flow with 1 step",
        "runtime_tools": [
            {
                "name": "get_place",
                "description": "Get place by place_uuid",
                "tool_type": "external",
                "parameters_schema": {
                    "type": "object",
                    "properties": {
                        "place_uuid": {
                            "type": "string",
                            "description": "UUID of the place to fetch"
                        }
                    },
                    "required": []
                },
                "config": {
                    "url": "https://api.example.com/get_place/{{place_uuid}}",
                    "method": "GET",
                    "headers": {
                        "Accept": "application/json",
                        "x-api-key": "your_api_key"
                    }
                }
            },
            {
                "name": "get_contact_profile",
                "description": "Retrieves or creates contact profile",
                "tool_type": "external",
                "parameters_schema": {
                    "type": "object",
                    "properties": {
                        "contact": {
                            "type": "object",
                            "properties": {
                                "email": {"type": "string"},
                                "first_name": {"type": "string"},
                                "last_name": {"type": "string"}
                            },
                            "required": ["email"]
                        }
                    },
                    "required": ["contact"]
                },
                "config": {
                    "url": "https://api.example.com/get_contact_profile",
                    "method": "POST",
                    "headers": {
                        "Accept": "application/json",
                        "x-api-key": "your_api_key"
                    }
                }
            }
        ],
        "max_tool_calls": 5,
        "tool_call_strategy": "auto"
    }
}
```

## Configuration

### Required Settings

- `api_key`: Your Travrse AI API key (Bearer token)

### Optional Settings

- `api_url`: API endpoint (default: `https://api.travrse.ai/v1/dispatch`)
- `model`: Model ID (default: `gpt-4o`)
- `response_format`: Output format - `text`, `json_object`, or `json_schema` (default: `text`)
- `stream_response`: Preference for streaming (default: `true`) - **Note**: Actual streaming is controlled by passing `queue` parameter to `ask_model()`
- `record_name`: Execution record name (default: `Standalone Execution`)
- `record_type`: Record type (default: `standalone`)
- `record_metadata`: Optional metadata object for the execution record
- `record_mode`: Record persistence - `virtual` or `persistent` (default: `virtual`)
- `flow_name`: Flow name (default: `Agent Flow`)
- `flow_description`: Flow description (default: `Flow with 1 step`)
- `flow_mode`: Flow persistence - `virtual` or `persistent` (default: `virtual`)
- `output_variable`: Name of output variable for step result (default: `prompt_result`)
- `step_name`: Name of the step in the flow (default: `Prompt 1`)
- `runtime_tools`: Array of external tool configurations
- `max_tool_calls`: Maximum tool calls allowed (default: `5`)
- `tool_call_strategy`: Tool calling strategy - `auto`, `required`, or `none` (default: `auto`)

See `configuration_schema.json` for the complete schema definition.

## Architecture

The module follows the AI Agent Handler pattern:

```
TravrseEventHandler (extends AIAgentEventHandler)
    ├── invoke_model()        # Makes API calls to Travrse AI
    ├── ask_model()           # Main entry point for requests
    ├── handle_response()     # Processes non-streaming responses
    ├── handle_stream()       # Processes streaming responses
    └── _cleanup_input_messages()  # Validates message sequences
```

### Key Components

1. **Message Cleanup**: Ensures valid conversation sequences before sending to API
2. **Timeline Tracking**: Monitors performance with millisecond precision
3. **Payload Building**: Constructs Travrse AI-compatible request payloads
4. **Response Handling**: Processes both streaming and non-streaming responses
5. **Error Recovery**: Automatic retries with exponential backoff

## Testing

Run the test suite:

```bash
python -m pytest travrse_agent_handler/tests/test_travrse_agent_handler.py -v
```

Or run tests directly:

```bash
python travrse_agent_handler/tests/test_travrse_agent_handler.py
```

The test suite includes:
- Basic non-streaming requests
- Streaming requests
- Requests with external tools

## Development

### Project Structure

```
travrse_agent_handler/
├── travrse_agent_handler/
│   ├── __init__.py
│   ├── travrse_agent_handler.py
│   └── tests/
│       └── test_travrse_agent_handler.py
├── configuration_schema.json
├── pyproject.toml
├── README.md
└── example.py
```

### Building

```bash
python -m build
```

### Contributing

1. Follow the existing code style
2. Add tests for new features
3. Update documentation
4. Ensure all tests pass

## Comparison with Ollama Handler

| Feature | Ollama Handler | Travrse Handler |
|---------|---------------|-----------------|
| Streaming | ✓ | ✓ |
| Tools/Functions | ✓ | ✓ |
| Multiple Models | Local only | Cloud-based |
| API Format | Ollama native | Travrse Flow API |
| Tool Execution | Client-side | Server-side or client-side |

## License

See LICENSE file for details.

## Author

IdeaBosque (ideabosque@gmail.com)

## Support

For issues and questions:
- GitHub Issues: [Create an issue](https://github.com/yourusername/travrse-agent-handler/issues)
- Email: ideabosque@gmail.com
