#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

__author__ = "bibow"

import json
import logging
import threading
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
        # Build messages array from input messages
        # Each message contains role and content fields for the Travrse AI API
        messages = []
        for msg in input_messages:
            messages.append(
                {"role": msg.get("role", ""), "content": msg.get("content", "")}
            )

        # Build the flow configuration
        step_config = {
            "user_prompt": "{{user_message}}",
            "system_prompt": self.agent.get("instructions", ""),
            "model": self.model_setting.get("model"),
            "response_format": self.output_format_type,
            "output_variable": self.model_setting.get(
                "output_variable", "prompt_result"
            ),
        }

        if "temperature" in self.model_setting:
            step_config["temperature"] = float(self.model_setting["temperature"])

        step_config["tools"] = {}
        if "tool_ids" in self.model_setting:
            step_config["tools"]["tool_ids"] = self.model_setting["tool_ids"]

        # Add tools if available - matching example.py structure
        runtime_tools = []
        if "tools" in self.model_setting:
            for tool in self.model_setting["tools"]:
                if tool["name"] not in self.model_setting.get("enabled_tools", []):
                    continue
                url = tool["config"]["url"]
                tool["config"]["url"] = url.format(endpoint_id=self.endpoint_id)
                runtime_tools.append(tool)
            step_config["tools"] = dict(
                step_config["tools"],
                **{
                    "runtime_tools": runtime_tools,
                    "max_tool_calls": self.model_setting.get("max_tool_calls", 5),
                    "tool_call_strategy": self.model_setting.get(
                        "tool_call_strategy", "auto"
                    ),
                },
            )

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
            "messages": messages,
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
        # Initialize variables at function scope for exception handler
        request_id = None
        payload = None
        stream = None

        try:
            invoke_start = pendulum.now("UTC")

            input_messages = kwargs.get("input_messages", [])
            stream = kwargs.get("stream", False)

            # Build Travrse AI payload
            payload = self._build_travrse_payload(input_messages)
            payload["options"]["stream_response"] = stream

            # Generate unique request ID for tracking
            request_id = f"req-{uuid.uuid4().hex[:12]}"

            # ===== API CALL TRACKING: Request Details =====
            self.logger.info(
                f"[API_CALL] Request ID: {request_id} | "
                f"URL: {self.api_url} | "
                f"Stream: {stream} | "
                f"Model: {payload['flow']['steps'][0]['config'].get('model')}"
            )

            # Validate payload has required fields
            if not payload.get("messages"):
                self.logger.warning(f"[API_CALL:{request_id}] No messages in payload")
            elif not any(msg.get("content") for msg in payload.get("messages", [])):
                self.logger.warning(f"[API_CALL:{request_id}] All messages have empty content")

            # Log message summary
            message_summary = []
            for i, msg in enumerate(payload.get("messages", [])):
                role = msg.get("role", "unknown")
                content_length = len(msg.get("content", ""))
                message_summary.append(f"{role}:{content_length}chars")

            self.logger.info(
                f"[API_CALL:{request_id}] Messages: [{', '.join(message_summary)}]"
            )

            # Log tools configuration
            step_config = payload['flow']['steps'][0]['config']
            if "tools" in step_config and step_config["tools"]:
                tool_ids = step_config["tools"].get("tool_ids", [])
                runtime_tools = step_config["tools"].get("runtime_tools", [])
                runtime_tool_names = [t.get("name", "unknown") for t in runtime_tools]

                self.logger.info(
                    f"[API_CALL:{request_id}] Tools - "
                    f"Built-in IDs: {tool_ids}, "
                    f"Runtime: {runtime_tool_names}, "
                    f"Max calls: {step_config['tools'].get('max_tool_calls', 0)}, "
                    f"Strategy: {step_config['tools'].get('tool_call_strategy', 'auto')}"
                )

            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    f"[API_CALL:{request_id}] Full payload: {Utility.json_dumps(payload)}"
                )

            # Make API request
            self.logger.info(f"[API_CALL:{request_id}] Sending request...")
            request_start = pendulum.now("UTC")

            # stream parameter must match stream_response in payload for proper handling
            response = requests.post(
                self.api_url,
                headers=self.headers,
                data=json.dumps(payload),
                stream=stream,
                timeout=120,  # 2 minute timeout
            )

            request_duration = (pendulum.now("UTC") - request_start).total_seconds() * 1000

            # ===== API CALL TRACKING: Response Details =====
            self.logger.info(
                f"[API_CALL:{request_id}] Response received in {request_duration:.2f}ms | "
                f"Status: {response.status_code} | "
                f"Content-Type: {response.headers.get('content-type', 'unknown')}"
            )

            if response.status_code != 200:
                error_text = response.text[:500] if response.text else "No error message"
                self.logger.error(
                    f"[API_CALL:{request_id}] FAILED - API request failed | "
                    f"Status: {response.status_code} | "
                    f"Error: {error_text}"
                )

                # Try to parse error details
                try:
                    error_json = response.json()
                    if "error" in error_json:
                        self.logger.error(
                            f"[API_CALL:{request_id}] Error details: {Utility.json_dumps(error_json['error'])}"
                        )
                except:
                    pass

                raise Exception(
                    f"API request failed with status {response.status_code}: {error_text}"
                )

            # Validate response content type for streaming
            content_type = response.headers.get("content-type", "")
            if stream:
                if "text/event-stream" not in content_type and "stream" not in content_type:
                    self.logger.warning(
                        f"[API_CALL:{request_id}] WARNING - Expected SSE stream but got content-type: {content_type}"
                    )
                else:
                    self.logger.info(
                        f"[API_CALL:{request_id}] SUCCESS - Valid streaming response (content-type: {content_type})"
                    )

            # Log response headers in debug mode
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    f"[API_CALL:{request_id}] Response headers: {dict(response.headers)}"
                )

            if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
                invoke_end = pendulum.now("UTC")
                invoke_time = (invoke_end - invoke_start).total_seconds() * 1000
                elapsed = self._get_elapsed_time()
                self.logger.info(
                    f"[TIMELINE] T+{elapsed:.2f}ms: API call returned (took {invoke_time:.2f}ms)"
                )

            # Store request_id on response object for correlation in handle_stream
            response.request_id = request_id

            self.logger.info(f"[API_CALL:{request_id}] SUCCESS - Request complete")

            return response

        except Exception as e:
            # Use variables from function scope
            req_id = request_id if request_id else 'unknown'
            self.logger.error(f"[API_CALL:{req_id}] FAILED - Error invoking model: {str(e)}")

            # Log additional context if available
            try:
                if payload and isinstance(payload, dict):
                    self.logger.error(
                        f"[API_CALL:{req_id}] Request context - "
                        f"Messages: {len(payload.get('messages', []))}, "
                        f"Stream: {stream}"
                    )
            except Exception:
                pass  # Don't fail on logging errors

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
            run_id = f"run-travrse-{self.model_setting['model']}-{timestamp}-{uuid.uuid4().hex[:8]}"

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

        # Get request_id from response object for correlation
        request_id = getattr(response, 'request_id', 'unknown')

        if retry_count > MAX_RETRIES:
            error_msg = (
                f"Maximum retry limit ({MAX_RETRIES}) exceeded for empty responses"
            )
            self.logger.error(f"[API_CALL:{request_id}] {error_msg}")
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
            message_id = f"msg-travrse-{self.model_setting['model']}-{timestamp}-{uuid.uuid4().hex[:8]}"

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

        Tracks stream processing with request_id for correlation.

        Travrse AI streams JSON objects with different types:
        - {"type":"step_chunk","text":"..."} - Text chunks to accumulate
        - {"type":"step_complete","result":{"response":"..."}} - Step completion with full response
        - {"type":"flow_complete",...} - Flow completion metadata

        Handles three scenarios:
        1. Empty stream → Retry up to 5 times
        2. Valid stream → Accumulate and set final_output
        3. Stream end → Send completion signal

        Args:
            response: Streaming HTTP response
            input_messages: Current conversation history
            stream_event: Event to signal completion
            retry_count: Current retry count (max 5 retries)

        Handles:
            - Accumulating response text
            - Processing JSON vs text formats
            - Sending chunks to websocket
            - Signaling completion
        """
        MAX_RETRIES = 5

        # Get request_id from response object for correlation
        request_id = getattr(response, 'request_id', 'unknown')
        stream_start = pendulum.now("UTC")

        # ===== API CALL TRACKING: Stream Processing Start =====
        self.logger.info(
            f"[API_CALL:{request_id}] Starting stream processing (attempt {retry_count + 1}/{MAX_RETRIES + 1})"
        )

        if retry_count > MAX_RETRIES:
            error_msg = f"Maximum retry limit ({MAX_RETRIES}) exceeded"
            self.logger.error(f"[API_CALL:{request_id}] {error_msg}")
            raise Exception(error_msg)

        message_id = None
        # Use list for efficient string concatenation (performance optimization)
        accumulated_text_parts = []
        accumulated_partial_json = ""
        accumulated_partial_text = ""
        received_any_content = False
        # Use cached output format type (performance optimization)
        output_format = self.output_format_type
        index = 0

        try:
            # Don't send initial stream data until we have actual content
            # Moved this after message_id is created to avoid early WebSocket failures

            # Process streaming response - each line is a JSON object
            step_result = None
            flow_metadata = None
            incomplete_line_buffer = ""  # Buffer for incomplete JSON lines
            line_count = 0
            chunk_types_received = {}  # Track what types of chunks we receive

            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    f"[API_CALL:{request_id}] Starting to read streaming response"
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

                    # Track chunk types for debugging
                    chunk_types_received[chunk_type] = chunk_types_received.get(chunk_type, 0) + 1

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
                            timestamp = pendulum.now("UTC").int_timestamp
                            message_id = f"msg-travrse-{self.model_setting['model']}-{timestamp}-{uuid.uuid4().hex[:8]}"

                            # Send initial stream message now that we have actual content
                            try:
                                self.send_data_to_stream(
                                    index=index,
                                    data_format=output_format,
                                )
                                index += 1
                            except Exception as stream_err:
                                # Log but don't fail - client may have disconnected
                                self.logger.warning(
                                    f"Failed to send initial stream data (client likely disconnected): {stream_err}"
                                )
                                # Continue processing to build final_output even if client disconnected

                        # Print and accumulate text
                        print(text_chunk, end="", flush=True)
                        accumulated_text_parts.append(text_chunk)

                        # Process content based on output format
                        if output_format in ["json_object", "json_schema"]:
                            accumulated_partial_json += text_chunk
                            # Temporarily build accumulated_text for processing
                            temp_accumulated_text = "".join(accumulated_text_parts)
                            index, temp_accumulated_text, accumulated_partial_json = (
                                self.process_and_send_json(
                                    index,
                                    temp_accumulated_text,
                                    accumulated_partial_json,
                                    output_format,
                                )
                            )
                        else:
                            accumulated_partial_text += text_chunk
                            # Check if text contains XML-style tags and update format
                            index, accumulated_partial_text = self.process_text_content(
                                index, accumulated_partial_text, output_format
                            )

                    elif chunk_type == "step_complete":
                        # Step completed - may contain full response
                        step_result = chunk_data.get("result", {})
                        step_success = chunk_data.get("success", False)
                        step_name = chunk_data.get("name", "Unknown")
                        step_error = chunk_data.get("error")

                        if self.logger.isEnabledFor(logging.INFO):
                            self.logger.info(
                                f"[handle_stream] Step '{step_name}' completed successfully: {step_success}"
                            )

                        # Check for step-level errors
                        if not step_success or step_error:
                            error_msg = step_error or "Step failed without error message"
                            self.logger.error(
                                f"[handle_stream] Step '{step_name}' failed: {error_msg}"
                            )
                            # Log full chunk_data for debugging
                            if self.logger.isEnabledFor(logging.DEBUG):
                                self.logger.debug(
                                    f"[handle_stream] Failed step data: {Utility.json_dumps(chunk_data)}"
                                )

                        # IMPORTANT: If no chunks received yet, check if response is in step_result
                        # Some models/configurations return the full response here instead of streaming
                        if not received_any_content and step_result:
                            if self.logger.isEnabledFor(logging.DEBUG):
                                self.logger.debug(
                                    f"[handle_stream] No chunks received yet. step_result keys: {list(step_result.keys())}"
                                )

                            # Try to extract response from step_result
                            # Common fields: 'response', 'output', 'content', 'text', output_variable name
                            response_text = None
                            output_var = self.model_setting.get("output_variable", "prompt_result")

                            # Check various possible locations for the response
                            if output_var in step_result:
                                response_text = step_result.get(output_var)
                            elif "response" in step_result:
                                response_text = step_result.get("response")
                            elif "output" in step_result:
                                response_text = step_result.get("output")
                            elif "content" in step_result:
                                response_text = step_result.get("content")
                            elif "text" in step_result:
                                response_text = step_result.get("text")

                            if response_text and isinstance(response_text, str) and response_text.strip():
                                self.logger.info(
                                    f"[handle_stream] Found response in step_result (length: {len(response_text)})"
                                )
                                received_any_content = True

                                if not message_id:
                                    timestamp = pendulum.now("UTC").int_timestamp
                                    message_id = f"msg-travrse-{self.model_setting['model']}-{timestamp}-{uuid.uuid4().hex[:8]}"

                                # Accumulate the response
                                accumulated_text_parts.append(response_text)

                                # Send to stream if possible
                                try:
                                    self.send_data_to_stream(
                                        index=index,
                                        data_format=output_format,
                                        chunk_delta=response_text,
                                    )
                                    index += 1
                                except Exception as stream_err:
                                    self.logger.warning(
                                        f"Failed to send step_result content to stream: {stream_err}"
                                    )
                            else:
                                self.logger.warning(
                                    f"[handle_stream] step_result present but no recognizable response found. Keys: {list(step_result.keys())}"
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
                                "[handle_stream] Skipping empty/whitespace line"
                            )
                    else:
                        # Real JSON parsing error - log and skip
                        incomplete_line_buffer = ""
                        if self.logger.isEnabledFor(logging.WARNING):
                            self.logger.warning(
                                f"[handle_stream] Failed to parse JSON: {full_line[:200]} - Error: {e}"
                            )
                    continue

            # Send any remaining partial text
            if len(accumulated_partial_text) > 0:
                try:
                    self.send_data_to_stream(
                        index=index,
                        data_format=output_format,
                        chunk_delta=accumulated_partial_text,
                    )
                    accumulated_partial_text = ""
                    index += 1
                except Exception as stream_err:
                    self.logger.warning(
                        f"Failed to send partial text to stream (client likely disconnected): {stream_err}"
                    )

            # ===== API CALL TRACKING: Stream Reading Complete =====
            stream_read_duration = (pendulum.now("UTC") - stream_start).total_seconds() * 1000
            self.logger.info(
                f"[API_CALL:{request_id}] Stream reading complete in {stream_read_duration:.2f}ms | "
                f"Lines: {line_count}, Content: {received_any_content}, "
                f"Chunks: {chunk_types_received}"
            )

            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    f"[API_CALL:{request_id}] Detailed stream summary - "
                    f"Lines: {line_count}, Content received: {received_any_content}, "
                    f"Chunk types: {chunk_types_received}"
                )

            # Build final accumulated text from parts (performance optimization)
            final_accumulated_text = "".join(accumulated_text_parts)

            # Scenario 1: Empty stream - retry or fail gracefully
            if not received_any_content or not final_accumulated_text.strip():
                if retry_count >= MAX_RETRIES:
                    # Max retries exceeded - set error message in final_output
                    error_msg = f"Maximum retry limit ({MAX_RETRIES}) exceeded - received empty stream"
                    self.logger.error(error_msg)

                    # Ensure final_output has valid structure even on failure
                    timestamp = pendulum.now("UTC").int_timestamp
                    self.final_output = {
                        "message_id": (
                            message_id
                            if message_id
                            else f"msg-error-{timestamp}-{uuid.uuid4().hex[:8]}"
                        ),
                        "role": "assistant",
                        "content": f"Error: {error_msg}. The model did not return any content after {MAX_RETRIES} attempts.",
                    }

                    # Send error message to stream if possible
                    try:
                        self.send_data_to_stream(
                            index=index,
                            data_format=output_format,
                            chunk_delta=self.final_output["content"],
                        )
                        index += 1
                        self.send_data_to_stream(
                            index=index,
                            data_format=output_format,
                            is_message_end=True,
                        )
                    except Exception as stream_err:
                        self.logger.warning(
                            f"Failed to send error message to stream: {stream_err}"
                        )
                    return

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

            # Scenario 2: Valid stream - finalize
            # Send final message end signal
            try:
                self.send_data_to_stream(
                    index=index,
                    data_format=output_format,
                    is_message_end=True,
                )
            except Exception as stream_err:
                self.logger.warning(
                    f"Failed to send final message end signal (client likely disconnected): {stream_err}"
                )

            # Build final output with metadata
            self.final_output = {
                "message_id": (
                    message_id
                    if message_id
                    # Optimized UUID generation
                    else f"msg-{pendulum.now('UTC').int_timestamp}-{uuid.uuid4().hex[:8]}"
                ),
                "role": "assistant",
                "content": final_accumulated_text,
            }

            # Add step result if available
            if step_result:
                self.final_output["step_result"] = step_result

            # Add flow metadata if available
            if flow_metadata:
                self.final_output["flow_metadata"] = flow_metadata

            # Store accumulated_text for backward compatibility
            self.accumulated_text = final_accumulated_text

            # ===== API CALL TRACKING: Stream Complete Success =====
            total_duration = (pendulum.now("UTC") - stream_start).total_seconds() * 1000
            content_length = len(final_accumulated_text)
            self.logger.info(
                f"[API_CALL:{request_id}] SUCCESS - Stream processing complete | "
                f"Total time: {total_duration:.2f}ms | "
                f"Content length: {content_length} chars | "
                f"Message ID: {self.final_output['message_id']}"
            )

            if flow_metadata:
                self.logger.info(
                    f"[API_CALL:{request_id}] Flow metadata - "
                    f"Execution ID: {flow_metadata.get('execution_id')}, "
                    f"Steps: {flow_metadata.get('successful_steps')}/{flow_metadata.get('total_steps')} successful"
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
            self.logger.error(f"[API_CALL:{request_id}] FAILED - Error in handle_stream: {str(e)}")
            raise
        finally:
            # Signal that streaming has finished
            if stream_event:
                stream_event.set()
