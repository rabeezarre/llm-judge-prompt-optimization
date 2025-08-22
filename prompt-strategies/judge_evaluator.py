#!/usr/bin/env python3
"""
Enhanced Judge Evaluation System with .env support
Evaluates chat completion events using judge definitions with dynamic output schemas.
"""

import json
import asyncio
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Using environment variables directly.")

from mistralai import Mistral


@dataclass
class OutputOption:
    """Represents a possible output value with description"""

    value: str
    description: str = ""


@dataclass
class JudgeDefinition:
    """Judge configuration for evaluating events"""

    name: str
    description: str
    model_name: str
    instructions: str
    output_options: List[OutputOption]
    tools: List[str] = None

    def __post_init__(self):
        if self.tools is None:
            self.tools = []

        if self.output_options and isinstance(self.output_options[0], dict):
            self.output_options = [OutputOption(**opt) for opt in self.output_options]


@dataclass
class ChatEvent:
    """Represents a chat completion event to be evaluated"""

    correlation_id: str = None
    correlationId: str = None  # Support both formats
    messages: List[Dict[str, Any]] = None
    model: str = None
    response: str = None
    metadata: Dict[str, Any] = None

    createdAt: str = None
    extraFields: Dict[str, Any] = None
    nbInputTokens: int = None
    nbOutputTokens: int = None
    nbMessages: int = None
    enabledTools: List[str] = None
    requestMessages: List[Dict[str, Any]] = None
    responseMessages: List[Dict[str, Any]] = None

    def __post_init__(self):
        if not self.correlation_id and self.correlationId:
            self.correlation_id = self.correlationId
        elif not self.correlationId and self.correlation_id:
            self.correlationId = self.correlation_id

        if self.metadata is None:
            self.metadata = {}
        if self.extraFields is None:
            self.extraFields = {}
        if self.enabledTools is None:
            self.enabledTools = []

        # Extract model from extraFields if not provided directly
        if not self.model and self.extraFields:
            self.model = self.extraFields.get("model_name", "unknown")

        # Build conversation from requestMessages and responseMessages if available
        if not self.messages:
            self.messages = []

            # Add request messages (usually system + user messages)
            if self.requestMessages:
                self.messages.extend(self.requestMessages)

            # Add response messages (assistant responses)
            if self.responseMessages:
                self.messages.extend(self.responseMessages)

        # Extract response content from responseMessages if not provided
        if not self.response and self.responseMessages:
            # Get the latest assistant response
            for msg in reversed(self.responseMessages):
                if msg.get("role") == "assistant" and msg.get("content"):
                    self.response = msg["content"]
                    break

        # Fallback: use last_user_message_preview if no proper messages
        if not self.messages and self.extraFields:
            last_message = self.extraFields.get("last_user_message_preview", "")
            if last_message:
                self.messages = [{"role": "user", "content": last_message}]
            else:
                self.messages = [
                    {"role": "user", "content": "No message content available"}
                ]

        if not self.response:
            self.response = "[No response content available]"


@dataclass
class JudgeResult:
    """Result of judge evaluation"""

    correlation_id: str
    judge_name: str
    answer: str
    analysis: str
    timestamp: str
    error: Optional[str] = None


class JudgeEvaluator:
    """Main class for evaluating events with judges"""

    def __init__(self, api_key: str, base_url: str = None):
        """Initialize with Mistral client"""
        self.client = Mistral(api_key=api_key, server_url=base_url)

    def _build_system_prompt(self, judge: JudgeDefinition, event: ChatEvent) -> str:
        """Build the system prompt for judge evaluation"""
        output_options_text = ""
        if judge.output_options:
            options_list = []
            for opt in judge.output_options:
                if opt.description:
                    options_list.append(f"- {opt.value}: {opt.description}")
                else:
                    options_list.append(f"- {opt.value}")
            output_options_text = f"\n\nOutput options:\n" + "\n".join(options_list)

        return f"""You are an expert judge evaluating a conversation.

{judge.instructions}

Please evaluate the following conversation and provide:
1. Your judgment as one of the specified output values
2. A brief analysis explaining your reasoning

{output_options_text}

Format your response as JSON:
{{
    "answer": "your_selected_output_value",
    "analysis": "your_reasoning_here"
}}"""

    def _build_conversation_context(self, event: ChatEvent) -> str:
        """Build conversation context from chat event"""
        context_parts = []

        # Add model info
        model = (
            event.model
            or (event.extraFields and event.extraFields.get("model_name"))
            or "unknown"
        )
        context_parts.append(f"Model: {model}")

        # Add token usage info if available
        if event.nbInputTokens or event.nbOutputTokens:
            token_info = []
            if event.nbInputTokens:
                token_info.append(f"Input: {event.nbInputTokens}")
            if event.nbOutputTokens:
                token_info.append(f"Output: {event.nbOutputTokens}")
            context_parts.append(f"Tokens: {', '.join(token_info)}")

        # Add full conversation
        if event.messages:
            context_parts.append("\nFull Conversation:")
            for i, msg in enumerate(event.messages):
                role = msg.get("role", "unknown").upper()
                content = msg.get("content", "")

                # Truncate very long system messages for readability
                if role == "SYSTEM" and len(content) > 500:
                    content = content[:500] + "... [truncated]"

                context_parts.append(f"{role}: {content}")

        # Add enabled tools if available
        if event.enabledTools:
            # Handle both list of strings and list of dicts
            if event.enabledTools and isinstance(event.enabledTools[0], dict):
                tool_names = [
                    tool.get("function", {}).get("name", "unknown")
                    for tool in event.enabledTools
                ]
                context_parts.append(f"\nEnabled Tools: {', '.join(tool_names)}")
            else:
                context_parts.append(
                    f"\nEnabled Tools: {', '.join(event.enabledTools)}"
                )

        # Add relevant metadata for context
        if event.extraFields:
            extra_context = []

            # Add detected language if available
            if "input_detected_language" in event.extraFields:
                extra_context.append(
                    f"Input Language: {event.extraFields['input_detected_language']}"
                )

            # Add tool information
            invoked_tools = event.extraFields.get("invoked_tools", [])
            if invoked_tools:
                extra_context.append(f"Tools Used: {', '.join(invoked_tools)}")

            # Add content flags
            content_flags = []
            if event.extraFields.get("has_code", False):
                content_flags.append("contains code")
            if event.extraFields.get("has_list", False):
                content_flags.append("contains list")
            if event.extraFields.get("has_image_in_conversation", False):
                content_flags.append("contains image")
            if event.extraFields.get("has_document_in_conversation", False):
                content_flags.append("contains document")

            if content_flags:
                extra_context.append(f"Content Features: {', '.join(content_flags)}")

            # Add performance info
            if "total_time_elapsed" in event.extraFields:
                extra_context.append(
                    f"Processing Time: {event.extraFields['total_time_elapsed']:.3f}s"
                )

            # Add refusal info if present
            if event.extraFields.get("has_refusal", False):
                refusal_reason = event.extraFields.get("refusal_reason", "Unknown")
                extra_context.append(f"Refusal: {refusal_reason}")

            if extra_context:
                context_parts.append(
                    f"\nAdditional Context:\n" + "\n".join(extra_context)
                )

        return "\n".join(context_parts)

    async def evaluate_event(
        self, judge: JudgeDefinition, event: ChatEvent
    ) -> JudgeResult:
        """Evaluate a single event with the given judge"""
        try:
            system_prompt = self._build_system_prompt(judge, event)
            conversation_context = self._build_conversation_context(event)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": conversation_context},
            ]

            # Prepare tools if specified
            tools = None
            if judge.tools and len(judge.tools) > 0:
                tools = []
                for tool_name in judge.tools:
                    if tool_name == "web_search":
                        tools.append({"type": "web_search"})
                    elif tool_name == "code_interpreter":
                        tools.append({"type": "code_interpreter"})

            # Make the API call
            response = await self.client.chat.complete_async(
                model=judge.model_name,
                messages=messages,
                tools=tools,
                response_format={"type": "json_object"}
                if not tools
                else None,  # Don't use JSON mode with tools
                temperature=0.1,  # Low temperature for consistent judgments
            )

            # Parse response
            content = response.choices[0].message.content

            # Try to parse as JSON first
            try:
                result_data = json.loads(content)
                answer = result_data.get("answer", "")
                analysis = result_data.get("analysis", "")
            except json.JSONDecodeError:
                # If not JSON, try to extract answer and analysis from text
                answer = ""
                analysis = content

                # Simple heuristic to extract answer if it matches output options
                content_lower = content.lower()
                for opt in judge.output_options:
                    if opt.value.lower() in content_lower:
                        answer = opt.value
                        break

            return JudgeResult(
                correlation_id=event.correlation_id or event.correlationId or "unknown",
                judge_name=judge.name,
                answer=answer,
                analysis=analysis,
                timestamp=datetime.now().isoformat(),
            )

        except Exception as e:
            return JudgeResult(
                correlation_id=event.correlation_id or event.correlationId or "unknown",
                judge_name=judge.name,
                answer="",
                analysis="",
                timestamp=datetime.now().isoformat(),
                error=str(e),
            )

    async def evaluate_batch(
        self, judge: JudgeDefinition, events: List[ChatEvent], max_concurrent: int = 5
    ) -> List[JudgeResult]:
        """Evaluate multiple events with the same judge with concurrency control"""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_evaluate(event):
            async with semaphore:
                return await self.evaluate_event(judge, event)

        tasks = [bounded_evaluate(event) for event in events]
        return await asyncio.gather(*tasks)


def load_judge_definition(file_path: str, use_folders: bool = False) -> JudgeDefinition:
    """Load judge definition from JSON file with optional folder organization"""
    if use_folders:
        judge_path = Path(file_path)
        if len(judge_path.parts) == 1:  # Just a filename
            judges_dir = Path("judges")
            if (judges_dir / file_path).exists():
                file_path = judges_dir / file_path

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return JudgeDefinition(**data)


def load_events(file_path: str, use_folders: bool = False) -> List[ChatEvent]:
    """Load chat events from JSON file with optional folder organization"""
    if use_folders:
        events_path = Path(file_path)
        if len(events_path.parts) == 1:  # Just a filename
            events_dir = Path("events")
            if (events_dir / file_path).exists():
                file_path = events_dir / file_path

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    events = []

    if isinstance(data, dict):
        events.append(ChatEvent(**data))
    elif isinstance(data, list):
        for item in data:
            events.append(ChatEvent(**item))
    else:
        raise ValueError(
            f"Invalid JSON format: expected dict or list, got {type(data)}"
        )

    return events


def save_results(
    results: List[JudgeResult],
    output_file: str,
    auto_unique: bool = False,
    use_folders: bool = False,
):
    """Save evaluation results to JSON file with optional unique naming and folder organization"""
    if use_folders:
        evaluations_dir = Path("evaluations")
        evaluations_dir.mkdir(exist_ok=True)

        output_path = Path(output_file)
        if len(output_path.parts) == 1:  # Just a filename, no path
            output_file = evaluations_dir / output_file
        else:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    if auto_unique:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = Path(output_file).stem
        extension = Path(output_file).suffix or ".json"
        output_dir = Path(output_file).parent

        unique_filename = f"{base_name}_{timestamp}{extension}"
        output_file = output_dir / unique_filename

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    results_data = [asdict(result) for result in results]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {output_file}")


def print_summary(results: List[JudgeResult]):
    """Print a summary of the evaluation results"""
    successful = sum(1 for r in results if not r.error)
    failed = len(results) - successful

    print(f"\n{'=' * 50}")
    print(f"EVALUATION SUMMARY")
    print(f"{'=' * 50}")
    print(f"Total events evaluated: {len(results)}")
    print(f"Successful evaluations: {successful}")
    print(f"Failed evaluations: {failed}")

    if successful > 0:
        answer_counts = {}
        for result in results:
            if not result.error and result.answer:
                answer_counts[result.answer] = answer_counts.get(result.answer, 0) + 1

        if answer_counts:
            print(f"\nAnswer distribution:")
            for answer, count in sorted(answer_counts.items()):
                percentage = (count / successful) * 100
                print(f"  {answer}: {count} ({percentage:.1f}%)")

    if failed > 0:
        print(f"\nErrors encountered:")
        error_counts = {}
        for result in results:
            if result.error:
                error_type = (
                    result.error.split(":")[0] if ":" in result.error else result.error
                )
                error_counts[error_type] = error_counts.get(error_type, 0) + 1

        for error_type, count in error_counts.items():
            print(f"  {error_type}: {count} occurrences")

    print(f"{'=' * 50}")


async def main():
    """Main function to run the evaluation"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate chat events with judge definitions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --judge example_judge.json --events example_events.json --output results.json
  %(prog)s --judge example_judge.json --events example_events.json --output results.json --use-folders
  %(prog)s --judge find_steps_judge.json --events example_events.json --output step_eval.json --use-folders --unique-name
  %(prog)s --judge judge.json --events events.json --output results.json --max-concurrent 10
        """,
    )

    parser.add_argument(
        "--judge", required=True, help="Path to judge definition JSON file"
    )
    parser.add_argument("--events", required=True, help="Path to events JSON file")
    parser.add_argument(
        "--output", required=True, help="Path to output results JSON file"
    )
    parser.add_argument(
        "--unique-name",
        action="store_true",
        help="Automatically generate unique filename with timestamp",
    )
    parser.add_argument(
        "--use-folders",
        action="store_true",
        help="Use organized folder structure (events/, judges/, evaluations/)",
    )
    parser.add_argument(
        "--api-key", help="Mistral API key (or set MISTRAL_API_KEY env var)"
    )
    parser.add_argument(
        "--base-url", help="Custom API base URL (or set MISTRAL_BASE_URL env var)"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Maximum concurrent evaluations (default: 5)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()

    api_key = args.api_key or os.getenv("MISTRAL_API_KEY")
    base_url = args.base_url or os.getenv("MISTRAL_BASE_URL")

    if not api_key:
        print("[X] Error: API key required!")
        print("   Use --api-key argument or set MISTRAL_API_KEY environment variable")
        print("   You can set it in your .env file like: MISTRAL_API_KEY=your_key_here")
        return 1

    judge_file = args.judge
    events_file = args.events

    if args.use_folders:
        if not Path(judge_file).exists():
            judges_path = Path("judges") / judge_file
            if judges_path.exists():
                judge_file = str(judges_path)

        if not Path(events_file).exists():
            events_path = Path("events") / events_file
            if events_path.exists():
                events_file = str(events_path)

    if not Path(judge_file).exists():
        print(f"[X] Error: Judge file not found: {judge_file}")
        if args.use_folders:
            print(f"   Looked in current directory and judges/ folder")
        return 1

    if not Path(events_file).exists():
        print(f"[X] Error: Events file not found: {events_file}")
        if args.use_folders:
            print(f"   Looked in current directory and events/ folder")
        return 1

    try:
        if args.verbose:
            print(f"[...] Loading judge definition from {judge_file}")
        judge = load_judge_definition(judge_file, args.use_folders)

        if args.verbose:
            print(f"[...] Loading events from {events_file}")
        events = load_events(events_file, args.use_folders)

        print(f" Judge: {judge.name}")
        print(f" Model: {judge.model_name}")
        print(f" Events to evaluate: {len(events)}")
        print(f" Max concurrent: {args.max_concurrent}")

        if judge.tools:
            print(f" Tools: {', '.join(judge.tools)}")

        evaluator = JudgeEvaluator(api_key, base_url)

        print(f"\n Starting evaluation...")
        start_time = datetime.now()

        results = await evaluator.evaluate_batch(judge, events, args.max_concurrent)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        print_summary(results)
        print(f" Total time: {duration:.2f} seconds")
        print(f" Average time per event: {duration / len(events):.2f} seconds")

        save_results(results, args.output, args.unique_name, args.use_folders)

        if args.unique_name:
            print(f"[OK] Results saved with unique timestamp filename")
        elif args.use_folders:
            print(f"[OK] Results saved in organized folder structure")
        else:
            print(f"[OK] Results saved to {args.output}")

        return 0

    except FileNotFoundError as e:
        print(f"[X] Error: File not found - {e}")
        return 1
    except json.JSONDecodeError as e:
        print(f"[X] Error: Invalid JSON format - {e}")
        return 1
    except Exception as e:
        print(f"[X] Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
