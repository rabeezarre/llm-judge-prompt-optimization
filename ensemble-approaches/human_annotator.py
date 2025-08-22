#!/usr/bin/env python3
"""
Human Annotation Tool for Judge Evaluations
Allows human reviewers to annotate AI judge evaluations with ground truth labels.
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import argparse


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
        # Convert dict output_options to OutputOption objects if needed
        if self.output_options and isinstance(self.output_options[0], dict):
            self.output_options = [OutputOption(**opt) for opt in self.output_options]


@dataclass
class ChatEvent:
    """Represents a chat completion event to be evaluated"""

    correlation_id: str = None
    correlationId: str = None
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
        # Handle different ID field names
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
            if self.requestMessages:
                self.messages.extend(self.requestMessages)
            if self.responseMessages:
                self.messages.extend(self.responseMessages)

        # Extract response content from responseMessages if not provided
        if not self.response and self.responseMessages:
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
class HumanAnnotation:
    """Human annotation for an event"""

    correlation_id: str
    judge_name: str
    expected_output: str
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class HumanAnnotator:
    """Interactive annotation tool"""

    def __init__(self, judge: JudgeDefinition, events: List[ChatEvent]):
        self.judge = judge
        self.events = events
        self.events_by_id = {event.correlation_id: event for event in events}
        self.annotations = []

    def format_conversation(self, event: ChatEvent) -> str:
        """Format conversation for display"""
        lines = []

        # Add model info
        model = event.model or "unknown"
        lines.append(f"Model: {model}")

        # Add token usage if available
        if event.nbInputTokens or event.nbOutputTokens:
            token_info = []
            if event.nbInputTokens:
                token_info.append(f"Input: {event.nbInputTokens}")
            if event.nbOutputTokens:
                token_info.append(f"Output: {event.nbOutputTokens}")
            lines.append(f"Tokens: {', '.join(token_info)}")

        lines.append("-" * 80)

        # Format messages
        if event.messages:
            for i, msg in enumerate(event.messages):
                role = msg.get("role", "unknown").upper()
                content = msg.get("content", "")

                # Truncate very long system messages
                if role == "SYSTEM" and len(content) > 1000:
                    content = (
                        content[:1000] + "\n... [truncated - showing first 1000 chars]"
                    )

                lines.append(f"\n[{role}]")
                lines.append(content)
                lines.append("-" * 40)

        # Add enabled tools if available
        if event.enabledTools:
            try:
                # Handle both list of strings and list of dicts
                if event.enabledTools and isinstance(event.enabledTools[0], dict):
                    tool_names = []
                    for tool in event.enabledTools:
                        if isinstance(tool, dict) and "function" in tool:
                            tool_name = tool.get("function", {}).get("name", "unknown")
                            tool_names.append(tool_name)
                        else:
                            tool_names.append(str(tool))
                    lines.append(f"\nEnabled Tools: {', '.join(tool_names)}")
                else:
                    lines.append(
                        f"\nEnabled Tools: {', '.join(str(t) for t in event.enabledTools)}"
                    )
            except (IndexError, TypeError):
                # Fallback if enabledTools has unexpected format
                lines.append(f"\nEnabled Tools: {str(event.enabledTools)}")

        # Add extra context
        if event.extraFields:
            extra_info = []

            # Language info
            if event.extraFields.get("input_detected_language"):
                extra_info.append(
                    f"Language: {event.extraFields['input_detected_language']}"
                )

            # Intent info
            intent_group = event.extraFields.get("intent_group")
            intent_category = event.extraFields.get("intent_category")
            intent_subcategory = event.extraFields.get("intent_subcategory")

            if intent_group or intent_category:
                intent_parts = []
                if intent_group:
                    intent_parts.append(intent_group)
                if intent_category:
                    intent_parts.append(intent_category)
                if intent_subcategory:
                    intent_parts.append(intent_subcategory)
                extra_info.append(f"Intent: {' > '.join(intent_parts)}")

            # Tools used
            invoked_tools = event.extraFields.get("invoked_tools", [])
            if invoked_tools:
                extra_info.append(f"Tools Used: {', '.join(invoked_tools)}")

            # Content flags
            flags = []
            if event.extraFields.get("has_code", False):
                flags.append("code")
            if event.extraFields.get("has_list", False):
                flags.append("list")
            if event.extraFields.get("has_image_in_conversation", False):
                flags.append("image")
            if event.extraFields.get("has_document_in_conversation", False):
                flags.append("document")
            if event.extraFields.get("has_refusal", False):
                refusal_reason = event.extraFields.get("refusal_reason", "unknown")
                flags.append(
                    f"refusal ({refusal_reason})" if refusal_reason else "refusal"
                )

            if flags:
                extra_info.append(f"Content: {', '.join(flags)}")

            # Performance info
            if event.extraFields.get("total_time_elapsed"):
                extra_info.append(
                    f"Time: {event.extraFields['total_time_elapsed']:.3f}s"
                )

            # System instruction following
            if event.extraFields.get("system_instruction_following"):
                extra_info.append(
                    f"System Following: {event.extraFields['system_instruction_following']}"
                )

            # Satisfaction
            if event.extraFields.get("satisfaction"):
                extra_info.append(f"Satisfaction: {event.extraFields['satisfaction']}")

            if extra_info:
                lines.append(f"\nContext: {' | '.join(extra_info)}")

        return "\n".join(str(line) for line in lines)

    def display_judge_info(self):
        """Display judge information"""
        print("=" * 80)
        print(f"JUDGE: {self.judge.name}")
        print("=" * 80)
        print(f"Description: {self.judge.description}")
        print(f"Model: {self.judge.model_name}")
        print("\nInstructions:")
        print(self.judge.instructions)

        print("\nOutput Options:")
        for i, opt in enumerate(self.judge.output_options, 1):
            print(f"  {i}. {opt.value}")
            if opt.description:
                print(f"     {opt.description}")

        if self.judge.tools:
            print(f"\nTools: {', '.join(self.judge.tools)}")

        print("=" * 80)

    def get_user_input(self, event: ChatEvent) -> HumanAnnotation:
        """Get annotation from user for a single event"""
        print(f"\nEvent ID: {event.correlation_id}")
        print("=" * 80)

        # Display conversation
        conversation = self.format_conversation(event)
        print(conversation)

        print("\n" + "=" * 80)
        print("ANNOTATION REQUIRED")
        print("=" * 80)

        # Show options
        print("Output options:")
        for i, opt in enumerate(self.judge.output_options, 1):
            print(f"  {i}. {opt.value}")
            if opt.description:
                print(f"     → {opt.description}")

        # Get user choice
        while True:
            try:
                print(
                    f"\nSelect output (1-{len(self.judge.output_options)}) or 'q' to quit, 's' to skip:"
                )
                choice = input("> ").strip().lower()

                if choice == "q":
                    return None
                elif choice == "s":
                    return HumanAnnotation(
                        correlation_id=event.correlation_id,
                        judge_name=self.judge.name,
                        expected_output="SKIPPED",
                    )

                choice_num = int(choice)
                if 1 <= choice_num <= len(self.judge.output_options):
                    selected_option = self.judge.output_options[choice_num - 1]

                    return HumanAnnotation(
                        correlation_id=event.correlation_id,
                        judge_name=self.judge.name,
                        expected_output=selected_option.value,
                    )
                else:
                    print(
                        f"Please enter a number between 1 and {len(self.judge.output_options)}"
                    )

            except ValueError:
                print("Please enter a valid number")
            except KeyboardInterrupt:
                print("\nExiting...")
                return None

    def annotate_all(self, start_from: int = 0) -> List[HumanAnnotation]:
        """Annotate all events interactively"""
        self.display_judge_info()

        print(
            f"\nStarting annotation from event {start_from + 1} of {len(self.events)}"
        )
        print("Commands: 'q' = quit, 's' = skip event")
        print("Press Enter to continue...")
        input()

        annotations = []

        for i, event in enumerate(self.events[start_from:], start_from):
            print(f"\n\nProgress: {i + 1}/{len(self.events)}")

            annotation = self.get_user_input(event)

            if annotation is None:  # User quit
                break

            annotations.append(annotation)

            # Auto-save every 5 annotations
            if len(annotations) % 5 == 0:
                temp_file = (
                    f"temp_annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )
                self.save_annotations(annotations, temp_file)
                print(f"Auto-saved to {temp_file}")

        return annotations

    def save_annotations(self, annotations: List[HumanAnnotation], output_file: str):
        """Save annotations to file"""
        annotations_data = [asdict(annotation) for annotation in annotations]

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(annotations_data, f, indent=2, ensure_ascii=False)


def load_judge_definition(file_path: str, use_folders: bool = False) -> JudgeDefinition:
    """Load judge definition from JSON file"""
    if use_folders:
        judge_path = Path(file_path)
        if len(judge_path.parts) == 1:
            judges_dir = Path("judges")
            if (judges_dir / file_path).exists():
                file_path = judges_dir / file_path

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return JudgeDefinition(**data)


def load_events(file_path: str, use_folders: bool = False) -> List[ChatEvent]:
    """Load chat events from JSON file"""
    if use_folders:
        events_path = Path(file_path)
        if len(events_path.parts) == 1:
            events_dir = Path("events")
            if (events_dir / file_path).exists():
                file_path = events_dir / file_path

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    events = []

    # Handle both single event object and array of events
    if isinstance(data, dict):
        # Single event object
        events.append(ChatEvent(**data))
    elif isinstance(data, list):
        # Array of events
        for item in data:
            events.append(ChatEvent(**item))
    else:
        raise ValueError(
            f"Invalid JSON format: expected dict or list, got {type(data)}"
        )

    return events


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Human annotation tool for judge evaluations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --judge instruction_following_judge.json --events example_events_wmsgs.json --use-folders
  %(prog)s --judge my_judge.json --events my_events.json --output human_annotations.json
  %(prog)s --judge judge.json --events events.json --start-from 10 --output annotations.json
        """,
    )

    parser.add_argument(
        "--judge", required=True, help="Path to judge definition JSON file"
    )
    parser.add_argument("--events", required=True, help="Path to events JSON file")
    parser.add_argument(
        "--output",
        help="Output file for annotations (default: annotations/annotations_TIMESTAMP.json)",
    )
    parser.add_argument(
        "--use-folders",
        action="store_true",
        help="Use organized folder structure (events/, judges/, annotations/)",
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=0,
        help="Start annotation from event number (0-based)",
    )

    args = parser.parse_args()

    # Validate input files
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
        return 1

    if not Path(events_file).exists():
        print(f"[X] Error: Events file not found: {events_file}")
        return 1

    try:
        # Load data
        print(f"[...] Loading judge definition from {judge_file}")
        judge = load_judge_definition(judge_file, args.use_folders)

        print(f"[...] Loading events from {events_file}")
        events = load_events(events_file, args.use_folders)

        print(f" Judge: {judge.name}")
        print(f" Events to annotate: {len(events)}")

        if args.start_from > 0:
            print(f"⏩ Starting from event {args.start_from + 1}")

        # Set up output file
        if args.output:
            output_file = args.output
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"annotations_{timestamp}.json"

        if args.use_folders:
            annotations_dir = Path("annotations")
            annotations_dir.mkdir(exist_ok=True)

            output_path = Path(output_file)
            if len(output_path.parts) == 1:  # Just filename
                output_file = annotations_dir / output_file

        # Ensure parent directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        # Start annotation
        annotator = HumanAnnotator(judge, events)
        annotations = annotator.annotate_all(args.start_from)

        if annotations:
            annotator.save_annotations(annotations, output_file)
            print(f"\n[OK] Saved {len(annotations)} annotations to {output_file}")

            # Print summary
            skipped = sum(1 for a in annotations if a.expected_output == "SKIPPED")
            annotated = len(annotations) - skipped

            print("\nSummary:")
            print(f"  Annotated: {annotated}")
            print(f"  Skipped: {skipped}")
            print(f"  Total: {len(annotations)}")

            if annotated > 0:
                output_counts = {}
                for annotation in annotations:
                    if annotation.expected_output != "SKIPPED":
                        output_counts[annotation.expected_output] = (
                            output_counts.get(annotation.expected_output, 0) + 1
                        )

                print("\nAnnotation distribution:")
                for output, count in sorted(output_counts.items()):
                    percentage = (count / annotated) * 100
                    print(f"  {output}: {count} ({percentage:.1f}%)")
        else:
            print("\n[X] No annotations completed")

        return 0

    except Exception as e:
        print(f"[X] Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
