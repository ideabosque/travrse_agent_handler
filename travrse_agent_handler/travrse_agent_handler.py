#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

__author__ = "bibow"

import json
import logging
import threading
import traceback
import uuid
from decimal import Decimal
from queue import Queue
from typing import Any, Dict, List, Optional

import pendulum
import requests

from ai_agent_handler import AIAgentEventHandler
from silvaengine_utility import Utility


# ----------------------------
# Travrse AI Response Streaming with Function Handling and History
# ----------------------------
class TravrseEventHandler(AIAgentEventHandler):
    """
    Manages conversations and function calls in real-time with Travrse AI:
      - Handles streaming responses from the model
      - Processes tool/function calls in responses
      - Executes functions and manages their lifecycle
      - Maintains conversation context and history
      - Handles both streaming and non-streaming responses
    """

    def __init__(
        self,
        logger: logging.Logger,
        agent: Dict[str, Any],
        **setting: Dict[str, Any],
    ) -> None:
        """
        Initialize the Travrse AI event handler

        Args:
            logger: Logger instance for debug/info messages
            agent: Configuration dict containing model instructions and settings
            setting: Additional settings for handler configuration
        """
        AIAgentEventHandler.__init__(self, logger, agent, **setting)

        # Enable timeline logging (default: False)
        self.enable_timeline_log = setting.get("enable_timeline_log", False)

        # Convert Decimal to float once during initialization (performance optimization)
        self.model_setting = {
            k: float(v) if isinstance(v, Decimal) else v
            for k, v in agent["configuration"].items()
        }

        # Cache frequently accessed configuration values
        self.api_url = self.model_setting.get(
            "api_url", "https://api.travrse.ai/v1/dispatch"
        )
        self.api_key = self.model_setting.get("api_key")

        # Cache output format type
        self.output_format_type = self.model_setting.get("response_format", "text")

        # Prepare headers for API requests
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _get_elapsed_time(self) -> float:
        """
        Get elapsed time in milliseconds from the first ask_model call.

        Returns:
            Elapsed time in milliseconds, or 0 if global start time not set
        """
        if not hasattr(self, "_global_start_time") or self._global_start_time is None:
            return 0.0
        return (pendulum.now("UTC") - self._global_start_time).total_seconds() * 1000

    def reset_timeline(self) -> None:
        """
        Reset the global timeline for a new run.
        Should be called at the start of each new user interaction/run.
        """
        self._global_start_time = None
        if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
            self.logger.info("[TIMELINE] Timeline reset for new run")

    def _build_travrse_payload(
        self, input_messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Build the Travrse AI API payload from input messages.
        Matches the exact format from example.py

        Args:
            input_messages: List of conversation messages

        Returns:
            Formatted payload for Travrse AI API
        """
        # Assemble user_prompt from all input messages
        # Format: role: content for each message
        user_prompt_parts = []
        for msg in input_messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if content:
                user_prompt_parts.append(f"{role}: {content}")

        user_prompt = "\n".join(user_prompt_parts) if user_prompt_parts else ""

        # Build the flow configuration
        step_config = {
            "user_prompt": user_prompt,
            "system_prompt": self.agent.get("instructions", ""),
            "model": self.model_setting.get("model", "gpt-4o"),
            "response_format": self.output_format_type,
            "output_variable": self.model_setting.get(
                "output_variable", "prompt_result"
            ),
        }

        # Add tools if available - matching example.py structure
        runtime_tools = []
        if "tools" in self.model_setting:
            for tool in self.model_setting["tools"]:
                if tool["name"] not in self.model_setting.get("enabled_tools", []):
                    continue
                url = tool["config"]["url"]
                tool["config"]["url"] = url.format(endpoint_id=self.endpoint_id)
                runtime_tools.append(tool)
            step_config["tools"] = {
                "runtime_tools": runtime_tools,
                "max_tool_calls": self.model_setting.get("max_tool_calls", 5),
                "tool_call_strategy": self.model_setting.get(
                    "tool_call_strategy", "auto"
                ),
            }

        # Build record section - matching example.py
        record = {
            "name": self.model_setting.get("record_name", "Standalone Execution"),
            "type": self.model_setting.get("record_type", "standalone"),
        }

        # Add metadata if provided
        if "record_metadata" in self.model_setting:
            record["metadata"] = self.model_setting["record_metadata"]

        payload = {
            "record": record,
            "flow": {
                "name": self.model_setting.get("flow_name", "Agent Flow"),
                "description": self.model_setting.get(
                    "flow_description", "Flow with 1 step"
                ),
                "steps": [
                    {
                        "id": f"step_{uuid.uuid4().hex}",
                        "name": self.model_setting.get("step_name", "Initial Prompt"),
                        "type": "prompt",
                        "enabled": True,
                        "config": step_config,
                    }
                ],
            },
            "options": {
                # stream_response is set in invoke_model based on the stream parameter
                "record_mode": self.model_setting.get("record_mode", "virtual"),
                "flow_mode": self.model_setting.get("flow_mode", "virtual"),
            },
        }

        return payload

    def invoke_model(self, **kwargs: Dict[str, Any]) -> Any:
        """
        Invokes the Travrse AI model with provided messages

        Args:
            kwargs: Contains input messages and streaming configuration

        Returns:
            Model response or streaming response

        Raises:
            Exception: If model invocation fails
        """
        try:
            invoke_start = pendulum.now("UTC")

            input_messages = kwargs.get("input_messages", [])
            stream = kwargs.get("stream", False)

            # Build Travrse AI payload
            payload = self._build_travrse_payload(input_messages)
            payload["options"]["stream_response"] = stream

            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    f"[invoke_model] Payload: {json.dumps(payload, indent=2)}"
                )

            # Make API request
            # stream parameter must match stream_response in payload for proper handling
            response = requests.post(
                self.api_url,
                headers=self.headers,
                data=json.dumps(payload),
                stream=stream,
            )

            if response.status_code != 200:
                raise Exception(
                    f"API request failed with status {response.status_code}: {response.text}"
                )

            if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
                invoke_end = pendulum.now("UTC")
                invoke_time = (invoke_end - invoke_start).total_seconds() * 1000
                elapsed = self._get_elapsed_time()
                self.logger.info(
                    f"[TIMELINE] T+{elapsed:.2f}ms: API call returned (took {invoke_time:.2f}ms)"
                )

            return response

        except Exception as e:
            if self.logger.isEnabledFor(logging.ERROR):
                self.logger.error(f"Error invoking model: {str(e)}")
            raise Exception(f"Failed to invoke model: {str(e)}")

    @Utility.performance_monitor.monitor_operation(operation_name="Travrse")
    def ask_model(
        self,
        input_messages: List[Dict[str, Any]],
        queue: Queue = None,
        stream_event: threading.Event = None,
        model_setting: Dict[str, Any] = None,
    ) -> Optional[str]:
        """
        Makes a request to the Travrse AI model with streaming or non-streaming mode

        Args:
            input_messages: List of conversation messages
            queue: Queue for streaming events
            stream_event: Event to signal streaming completion
            model_setting: Optional model configuration overrides

        Returns:
            Response ID for non-streaming requests, None for streaming

        Raises:
            Exception: If request processing fails
        """
        ask_model_start = pendulum.now("UTC")

        if not hasattr(self, "_ask_model_depth"):
            self._ask_model_depth = 0

        self._ask_model_depth += 1
        is_top_level = self._ask_model_depth == 1

        if is_top_level:
            self._global_start_time = ask_model_start
            if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
                self.logger.info("[TIMELINE] T+0ms: Run started - First ask_model call")
        else:
            if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
                elapsed = self._get_elapsed_time()
                self.logger.info(
                    f"[TIMELINE] T+{elapsed:.2f}ms: Recursive ask_model call started"
                )

        try:
            stream = True if queue is not None else False

            if model_setting:
                self.model_setting.update(model_setting)

            cleanup_start = pendulum.now("UTC")
            cleanup_end = pendulum.now("UTC")
            cleanup_time = (cleanup_end - cleanup_start).total_seconds() * 1000

            if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
                preparation_end = pendulum.now("UTC")
                preparation_time = (
                    preparation_end - ask_model_start
                ).total_seconds() * 1000
                elapsed = self._get_elapsed_time()
                self.logger.info(
                    f"[TIMELINE] T+{elapsed:.2f}ms: Preparation complete (took {preparation_time:.2f}ms, cleanup: {cleanup_time:.2f}ms)"
                )

            timestamp = pendulum.now("UTC").int_timestamp
            run_id = f"run-travrse-{self.model_setting.get('model', 'default')}-{timestamp}-{uuid.uuid4().hex[:8]}"

            response = self.invoke_model(
                **{
                    "input_messages": input_messages,
                    "stream": stream,
                }
            )

            if stream:
                queue.put({"name": "run_id", "value": run_id})
                self.handle_stream(response, input_messages, stream_event=stream_event)
                return None

            self.handle_response(response, input_messages)
            return run_id
        except Exception as e:
            self.logger.error(f"Error in ask_model: {str(e)}")
            raise Exception(f"Failed to process model request: {str(e)}")
        finally:
            self._ask_model_depth -= 1

            if self._ask_model_depth == 0:
                if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
                    elapsed = self._get_elapsed_time()
                    self.logger.info(
                        f"[TIMELINE] T+{elapsed:.2f}ms: Run complete - Resetting timeline"
                    )
                self._global_start_time = None

    def handle_response(
        self,
        response: requests.Response,
        input_messages: List[Dict[str, Any]],
        retry_count: int = 0,
    ) -> None:
        """
        Processes non-streaming model output from Travrse AI.

        Args:
            response: HTTP response object
            input_messages: Current conversation history
            retry_count: Current retry count (max 5 retries)
        """
        MAX_RETRIES = 5
        if retry_count > MAX_RETRIES:
            error_msg = (
                f"Maximum retry limit ({MAX_RETRIES}) exceeded for empty responses"
            )
            self.logger.error(error_msg)
            raise Exception(error_msg)

        try:
            # Travrse AI returns plain text response directly
            response_text = response.text

            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    f"[handle_response] Raw response: {response_text[:200]}..."
                )

            # Parse response
            if not response_text or not response_text.strip():
                self.logger.warning(
                    f"Received empty response from model, retrying (attempt {retry_count + 1}/{MAX_RETRIES})..."
                )
                next_response = self.invoke_model(
                    **{"input_messages": input_messages, "stream": False}
                )
                self.handle_response(
                    next_response, input_messages, retry_count=retry_count + 1
                )
                return

            # Set final output
            timestamp = pendulum.now("UTC").int_timestamp
            message_id = f"msg-travrse-{self.model_setting.get('model', 'default')}-{timestamp}-{uuid.uuid4().hex[:8]}"

            self.final_output = {
                "message_id": message_id,
                "role": "assistant",
                "content": response_text,
            }

            if self.logger.isEnabledFor(logging.INFO):
                self.logger.info(
                    f"[handle_response] Final output set with {len(response_text)} chars"
                )

        except Exception as e:
            self.logger.error(f"Error in handle_response: {str(e)}")
            raise

    def handle_stream(
        self,
        response: requests.Response,
        input_messages: List[Dict[str, Any]],
        stream_event: threading.Event = None,
        retry_count: int = 0,
    ) -> None:
        """
        Processes streaming model responses chunk by chunk from Travrse AI.

        Travrse AI streams JSON objects with different types:
        - {"type":"step_chunk","text":"..."} - Text chunks to accumulate
        - {"type":"step_complete","result":{"response":"..."}} - Step completion with full response
        - {"type":"flow_complete",...} - Flow completion metadata

        Args:
            response: Streaming HTTP response
            input_messages: Current conversation history
            stream_event: Event to signal completion
            retry_count: Current retry count (max 5 retries)
        """
        MAX_RETRIES = 5
        if retry_count > MAX_RETRIES:
            error_msg = f"Maximum retry limit ({MAX_RETRIES}) exceeded"
            self.logger.error(error_msg)
            raise Exception(error_msg)

        message_id = None
        accumulated_text_parts = []
        received_any_content = False
        output_format = self.output_format_type
        index = 0

        try:
            self.send_data_to_stream(
                index=index,
                data_format=output_format,
            )
            index += 1

            # Process streaming response - each line is a JSON object
            step_result = None
            flow_metadata = None
            incomplete_line_buffer = ""  # Buffer for incomplete JSON lines
            line_count = 0

            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    f"[handle_stream] Starting to read streaming response"
                )

            for line in response.iter_lines(decode_unicode=True, chunk_size=None):
                line_count += 1
                if not line:
                    if self.logger.isEnabledFor(logging.DEBUG):
                        self.logger.debug(
                            f"[handle_stream] Line {line_count}: Empty (None)"
                        )
                    continue

                # iter_lines with decode_unicode=True returns strings, not bytes
                line_str = (
                    line.strip()
                    if isinstance(line, str)
                    else line.decode("utf-8").strip()
                )

                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(
                        f"[handle_stream] Line {line_count}: '{line_str[:100]}...' (len={len(line_str)})"
                    )

                # Skip empty lines
                if not line_str:
                    continue

                # Travrse AI uses Server-Sent Events (SSE) format with "data: " prefix
                # Strip the prefix to get the JSON payload
                if line_str.startswith("data: "):
                    line_str = line_str[6:]  # Remove "data: " prefix

                # Skip if nothing left after stripping prefix
                if not line_str:
                    continue

                # Combine with any buffered incomplete line
                full_line = incomplete_line_buffer + line_str

                try:
                    # Try to parse the complete line as JSON
                    chunk_data = json.loads(full_line)
                    # Success - clear the buffer
                    incomplete_line_buffer = ""
                    chunk_type = chunk_data.get("type")

                    if self.logger.isEnabledFor(logging.DEBUG):
                        self.logger.debug(
                            f"[handle_stream] Received chunk type: {chunk_type}"
                        )

                    # Handle different chunk types
                    if chunk_type == "step_chunk":
                        # Extract text from step_chunk
                        text_chunk = chunk_data.get("text", "")

                        if not text_chunk:
                            continue

                        received_any_content = True

                        if not message_id:
                            self.send_data_to_stream(
                                index=index,
                                data_format=output_format,
                            )
                            index += 1

                            timestamp = pendulum.now("UTC").int_timestamp
                            message_id = f"msg-travrse-{self.model_setting.get('model', 'default')}-{timestamp}-{uuid.uuid4().hex[:8]}"

                        # Print and accumulate text
                        print(text_chunk, end="", flush=True)
                        accumulated_text_parts.append(text_chunk)

                        # Send to stream
                        self.send_data_to_stream(
                            index=index,
                            data_format=output_format,
                            chunk_delta=text_chunk,
                        )
                        index += 1

                    elif chunk_type == "step_complete":
                        # Step completed - may contain full response
                        step_result = chunk_data.get("result", {})
                        if self.logger.isEnabledFor(logging.INFO):
                            self.logger.info(
                                f"[handle_stream] Step '{chunk_data.get('name')}' completed successfully: {chunk_data.get('success')}"
                            )

                    elif chunk_type == "flow_complete":
                        # Flow completed - contains execution metadata
                        flow_metadata = {
                            "execution_id": chunk_data.get("executionId"),
                            "flow_id": chunk_data.get("flow_id"),
                            "flow_name": chunk_data.get("flowName"),
                            "execution_time": chunk_data.get("executionTime"),
                            "total_steps": chunk_data.get("totalSteps"),
                            "successful_steps": chunk_data.get("successfulSteps"),
                            "failed_steps": chunk_data.get("failedSteps"),
                        }
                        if self.logger.isEnabledFor(logging.INFO):
                            self.logger.info(
                                f"[handle_stream] Flow '{flow_metadata['flow_name']}' completed in {flow_metadata['execution_time']}ms"
                            )

                    else:
                        # Log unknown chunk types for debugging
                        if self.logger.isEnabledFor(logging.DEBUG):
                            self.logger.debug(
                                f"[handle_stream] Unknown chunk type: {chunk_type}"
                            )

                except json.JSONDecodeError as e:
                    # Handle incomplete JSON - might be truncated across multiple lines
                    error_msg = str(e)

                    # Check if this looks like an incomplete JSON object
                    if any(
                        indicator in error_msg
                        for indicator in [
                            "Unterminated string",
                            "Expecting property name",
                            "Expecting ',' delimiter",
                            "Expecting ':' delimiter",
                        ]
                    ):
                        # Likely incomplete - buffer it and wait for next line
                        incomplete_line_buffer = full_line
                        if self.logger.isEnabledFor(logging.DEBUG):
                            self.logger.debug(
                                f"[handle_stream] Buffering incomplete JSON line (length: {len(full_line)})"
                            )
                        continue
                    elif "Expecting value" in error_msg and "char 0" in error_msg:
                        # Empty line - already handled above, but just in case
                        incomplete_line_buffer = ""
                        if self.logger.isEnabledFor(logging.DEBUG):
                            self.logger.debug(
                                f"[handle_stream] Skipping empty/whitespace line"
                            )
                    else:
                        # Real JSON parsing error - log and skip
                        incomplete_line_buffer = ""
                        if self.logger.isEnabledFor(logging.WARNING):
                            self.logger.warning(
                                f"[handle_stream] Failed to parse JSON: {full_line[:200]} - Error: {e}"
                            )
                    continue

            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    f"[handle_stream] Finished reading stream. Total lines: {line_count}, Content received: {received_any_content}"
                )

            final_accumulated_text = "".join(accumulated_text_parts)

            if not received_any_content or not final_accumulated_text.strip():
                self.logger.warning(
                    f"Received empty stream from model (lines: {line_count}, content: {received_any_content}), retrying (attempt {retry_count + 1}/{MAX_RETRIES})..."
                )
                next_response = self.invoke_model(
                    **{"input_messages": input_messages, "stream": True}
                )
                self.handle_stream(
                    next_response, input_messages, stream_event, retry_count + 1
                )
                return

            # Build final output with metadata
            self.final_output = {
                "message_id": message_id,
                "role": "assistant",
                "content": final_accumulated_text,
            }

            # Add step result if available
            if step_result:
                self.final_output["step_result"] = step_result

            # Add flow metadata if available
            if flow_metadata:
                self.final_output["flow_metadata"] = flow_metadata

            self.send_data_to_stream(
                index=index,
                data_format=output_format,
            )

            if self._run is None:
                self._short_term_memory.append(
                    {
                        "message": {
                            "role": "assistant",
                            "content": final_accumulated_text,
                        },
                        "created_at": pendulum.now("UTC"),
                    }
                )

        except Exception as e:
            self.logger.error(f"Error in handle_stream: {str(e)}")
            raise
        finally:
            if stream_event:
                stream_event.set()
