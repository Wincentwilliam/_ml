"""
example_integration.py - Shows how to integrate v3_agent_secure with your agent0.py

This example demonstrates:
1. Basic integration with an OpenAI-style LLM client
2. Custom action handling
3. Security configuration options
"""

import os
import tempfile
from pathlib import Path
from v3_agent_secure import (
    SecureAgentWrapper,
    PathValidator,
    HumanInTheLoop,
    LLMSecurityReviewer,
    ActionPlan,
    SecurityDecision
)


def example_basic_usage():
    """Basic integration example."""
    
    # Create temp directory for testing
    test_dir = tempfile.mkdtemp(prefix="v3_secure_test_")
    Path(test_dir).joinpath("src").mkdir()
    Path(test_dir).joinpath("logs").mkdir()
    Path(test_dir).joinpath("src/main.py").write_text("# Main app\nprint('hello')")
    
    # Your LLM client function (adjust to your actual client)
    def llm_client(model: str, messages: list, temperature: float = 0.7):
        # Example with OpenAI:
        # return openai_client.chat.completions.create(
        #     model=model,
        #     messages=messages,
        #     temperature=temperature
        # )
        
        # For demo, return a mock response:
        return "<response>safe</response><reason>Mock review - all clear</reason>"
    
    # Create the secure wrapper
    wrapper = SecureAgentWrapper(
        base_dir=test_dir,                  # Restrict to this directory
        llm_client=llm_client,              # Your LLM client function
        llm_model="gpt-4",                 # Model for security reviews
        enable_llm_review=True,            # Enable LLM security layer
        enable_hitl=False,                 # Disable HITL for demo (use True for interactive)
        hitl_auto_approve_patterns=[       # Patterns that auto-approve
            r"test.*file",
            r"\.tmp$",
            r"logs/"
        ]
    )
    
    # Use wrapper for file operations
    print("=== File Operations ===")
    
    # Safe operation (within base_dir)
    content = wrapper.read_file(f"{test_dir}/src/main.py")
    if content:
        print(f"Read {len(content)} characters from main.py")
    
    # Unsafe operation (outside base_dir) - will be blocked
    content = wrapper.read_file("/etc/passwd")
    if content is None:
        print("Blocked: Path outside BASE_DIR")
    
    # Write operation
    success = wrapper.write_file(f"{test_dir}/output/results.txt", "Hello World")
    print(f"Write operation: {'Success' if success else 'Failed'}")
    
    # Get security stats
    stats = wrapper.get_stats()
    print(f"\n=== Security Stats ===")
    print(f"Total actions: {stats['total_actions']}")
    print(f"Allowed: {stats['allowed']}")
    print(f"Blocked: {stats['blocked']}")
    print(f"HITL overrides: {stats['hitl_overrides']}")
    
    # Cleanup
    import shutil
    shutil.rmtree(test_dir, ignore_errors=True)


def example_standalone_components():
    """Use components individually if needed."""
    
    # Create temp directory for testing
    test_dir = tempfile.mkdtemp(prefix="v3_secure_test_")
    Path(test_dir).joinpath("src").mkdir()
    Path(test_dir).joinpath("logs").mkdir()
    
    # PathValidator only
    validator = PathValidator(test_dir)
    
    is_safe, error = validator.validate_path(f"{test_dir}/src/app.py")
    print(f"Path validation (within base): {is_safe}")  # True
    
    is_safe, error = validator.validate_path("/etc/passwd")
    print(f"Path validation (system file): {is_safe}")  # False
    
    # HumanInTheLoop only
    hitl = HumanInTheLoop(
        enabled=False,  # Disabled for demo
        auto_approve_patterns=[r"logs/"]
    )
    
    action = ActionPlan(action_type="read_file", target_path="logs/app.log")
    
    # Will auto-approve (matches pattern)
    approved = hitl.should_auto_approve("Reading logs/app.log")
    print(f"Auto-approve logs: {approved}")
    
    # Will need manual approval
    approved = hitl.should_auto_approve("Reading /etc/passwd")
    print(f"Auto-approve system file: {approved}")
    
    # LLMSecurityReviewer only
    def mock_llm(**kwargs):
        return "<response>safe</response><reason>No security concerns</reason>"
    
    reviewer = LLMSecurityReviewer(
        llm_client=mock_llm,
        model="gpt-4"
    )
    
    action = ActionPlan(
        action_type="write_file",
        target_path="src/config.py",
        content="# Configuration"
    )
    
    review = reviewer.review_action(action, test_dir)
    print(f"LLM Review - Safe: {review.is_safe}")
    print(f"LLM Review - Reason: {review.reason}")
    
    # Cleanup
    import shutil
    shutil.rmtree(test_dir, ignore_errors=True)


def example_custom_action_handler():
    """Handle custom actions beyond basic file operations."""
    
    test_dir = tempfile.mkdtemp(prefix="v3_secure_test_")
    
    def llm_client(**kwargs):
        return "<response>safe</response><reason>OK</reason>"
    
    wrapper = SecureAgentWrapper(
        base_dir=test_dir,
        llm_client=llm_client,
        enable_hitl=False  # Disable for demo
    )
    
    # Create custom action plan
    action = ActionPlan(
        action_type="database_query",
        command="SELECT * FROM users WHERE id = 1"
    )
    
    # Manual review
    review = wrapper.review_action(action)
    print(f"Custom action safe: {review.is_safe}")
    
    # Process through security layers
    should_proceed, reason = wrapper._should_proceed(action)
    print(f"Proceed: {should_proceed}, Reason: {reason}")
    
    # Cleanup
    import shutil
    shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    print("=== Basic Usage Example ===")
    example_basic_usage()
    
    print("\n" + "=" * 60)
    print("=== Standalone Components Example ===")
    example_standalone_components()
    
    print("\n" + "=" * 60)
    print("=== Custom Action Handler Example ===")
    example_custom_action_handler()
