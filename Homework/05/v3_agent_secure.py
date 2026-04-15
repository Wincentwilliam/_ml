"""
v3_agent_secure.py - Security Layer for AI Agent

Provides three security features:
1. File Sandbox - Restricts file operations to BASE_DIR
2. Human-in-the-Loop - User approval for flagged operations
3. LLM Security Reviewer - Secondary LLM audits action plans

Usage:
    from v3_agent_secure import SecureAgentWrapper
    
    agent = SecureAgentWrapper(
        base_dir="./my_project",
        llm_client=your_llm_client,
        llm_model="gpt-4"
    )
"""

import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Tuple, Callable, Any
from dataclasses import dataclass
from enum import Enum


class SecurityDecision(Enum):
    ALLOW = "allow"
    DENY = "deny"
    ASK_USER = "ask_user"


@dataclass
class SecurityReview:
    """Result from LLM security reviewer."""
    is_safe: bool
    reason: str
    raw_response: str


@dataclass
class ActionPlan:
    """Represents an action the agent wants to perform."""
    action_type: str  # e.g., "read_file", "write_file", "execute"
    target_path: Optional[str] = None
    content: Optional[str] = None
    command: Optional[str] = None
    
    def to_review_string(self) -> str:
        """Convert action to string for LLM review."""
        parts = [f"Action: {self.action_type}"]
        if self.target_path:
            parts.append(f"Target: {self.target_path}")
        if self.content:
            content_preview = self.content[:200] + "..." if len(self.content) > 200 else self.content
            parts.append(f"Content preview: {content_preview}")
        if self.command:
            parts.append(f"Command: {self.command}")
        return "\n".join(parts)


class PathValidator:
    """Validates file paths to ensure they stay within BASE_DIR."""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir).resolve()
        if not self.base_dir.exists():
            raise ValueError(f"BASE_DIR does not exist: {self.base_dir}")
    
    def is_within_base_dir(self, path: str) -> bool:
        """Check if a path is within the allowed BASE_DIR."""
        try:
            resolved_path = Path(path).resolve()
            return resolved_path.is_relative_to(self.base_dir)
        except (ValueError, OSError):
            return False
    
    def validate_path(self, path: str, action: str = "access") -> Tuple[bool, str]:
        """
        Validate a path and return (is_valid, error_message).
        
        Returns:
            Tuple of (is_valid, error_message)
            If valid, error_message is empty string.
        """
        if not self.is_within_base_dir(path):
            return False, (
                f"Path '{path}' is outside the allowed directory '{self.base_dir}'. "
                f"Only files within '{self.base_dir}' can be {action}ed."
            )
        return True, ""
    
    def get_safe_path(self, path: str) -> Optional[Path]:
        """
        Resolve and validate a path, returning None if invalid.
        """
        try:
            resolved = Path(path).resolve()
            if self.is_within_base_dir(path):
                return resolved
            return None
        except (ValueError, OSError):
            return None


class HumanInTheLoop:
    """Handles user approval for flagged operations."""
    
    def __init__(self, enabled: bool = True, auto_approve_patterns: Optional[list] = None):
        self.enabled = enabled
        self.auto_approve_patterns = auto_approve_patterns or []
        self.approval_history: list[Tuple[str, bool]] = []
    
    def should_auto_approve(self, reason: str) -> bool:
        """Check if reason matches auto-approve patterns."""
        for pattern in self.auto_approve_patterns:
            if re.search(pattern, reason, re.IGNORECASE):
                return True
        return False
    
    def request_approval(self, action_plan: ActionPlan, reason: str) -> bool:
        """
        Request user approval for an action.
        
        Returns:
            True if user approves, False otherwise.
        """
        if not self.enabled:
            return False
        
        if self.should_auto_approve(reason):
            print(f"[AUTO-APPROVED] {reason}")
            return True
        
        print("\n" + "=" * 60)
        print("SECURITY INTERCEPTION - Human-in-the-Loop Required")
        print("=" * 60)
        print(f"\nAction: {action_plan.action_type}")
        if action_plan.target_path:
            print(f"Target: {action_plan.target_path}")
        if action_plan.command:
            print(f"Command: {action_plan.command}")
        print(f"\nReason: {reason}")
        print("-" * 60)
        
        while True:
            response = input("\nApprove this action? (y/n): ").strip().lower()
            if response in ('y', 'yes'):
                self.approval_history.append((action_plan.action_type, True))
                return True
            elif response in ('n', 'no'):
                self.approval_history.append((action_plan.action_type, False))
                return False
            else:
                print("Please enter 'y' or 'n'.")


class LLMSecurityReviewer:
    """
    Secondary LLM layer that audits action plans before execution.
    Expects XML format responses: <response>safe/unsafe</response><reason>...</reason>
    """
    
    SECURITY_PROMPT_TEMPLATE = """You are a security auditor for an AI coding agent.

Analyze the following action plan and determine if it poses any security risk.

Security rules:
1. File operations are restricted to: {base_dir}
2. Be suspicious of: path traversal attempts, system file access, secret exposure
3. Consider: Does this action access sensitive files (.env, credentials, keys)?
4. Consider: Does this action modify critical system files?

Action Plan:
{action_plan}

Respond ONLY in this exact XML format (no other text):
<response>safe</response>
<reason>Brief explanation of why this is safe</reason>

OR:

<response>unsafe</response>
<reason>Explanation of why this is unsafe and what risk it poses</reason>
"""
    
    def __init__(
        self,
        llm_client: Callable,
        model: str = "gpt-4",
        review_threshold: float = 0.5
    ):
        self.llm_client = llm_client
        self.model = model
        self.review_threshold = review_threshold
        self.review_history: list[SecurityReview] = []
    
    def review_action(self, action_plan: ActionPlan, base_dir: str) -> SecurityReview:
        """
        Send action plan to LLM for security review.
        
        Args:
            action_plan: The action to review
            base_dir: The allowed base directory
            
        Returns:
            SecurityReview object with is_safe, reason, and raw_response
        """
        prompt = self.SECURITY_PROMPT_TEMPLATE.format(
            base_dir=base_dir,
            action_plan=action_plan.to_review_string()
        )
        
        try:
            response = self.llm_client(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a strict security auditor."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            raw_response = response if isinstance(response, str) else response.get("content", "")
            
        except Exception as e:
            raw_response = f"<response>unsafe</response><reason>LLM review failed: {str(e)}</reason>"
        
        return self._parse_response(raw_response)
    
    def _parse_response(self, xml_response: str) -> SecurityReview:
        """Parse XML response from LLM."""
        try:
            wrapped = f"<root>{xml_response}</root>"
            root = ET.fromstring(wrapped)
            
            response_elem = root.find("response")
            reason_elem = root.find("reason")
            
            if response_elem is None or reason_elem is None:
                return SecurityReview(
                    is_safe=False,
                    reason="Failed to parse LLM response - using safe default",
                    raw_response=xml_response
                )
            
            response_text = response_elem.text.lower().strip() if response_elem.text else "unsafe"
            is_safe = "safe" in response_text and "unsafe" not in response_text
            reason = reason_elem.text or "No reason provided"
            
            review = SecurityReview(
                is_safe=is_safe,
                reason=reason,
                raw_response=xml_response
            )
            self.review_history.append(review)
            return review
            
        except ET.ParseError:
            if "safe" in xml_response.lower():
                return SecurityReview(
                    is_safe=True,
                    reason="Response parsed heuristically as safe",
                    raw_response=xml_response
                )
            return SecurityReview(
                is_safe=False,
                reason="Failed to parse XML response - blocked by default",
                raw_response=xml_response
            )


class SecureAgentWrapper:
    """
    Main wrapper class that combines all security features.
    
    Usage:
        wrapper = SecureAgentWrapper(
            base_dir="./my_project",
            llm_client=openai_client.chat.completions.create,
            llm_model="gpt-4"
        )
        
        # Use wrapper for file operations
        content = wrapper.read_file("src/app.py")
        wrapper.write_file("output.txt", "Hello")
    """
    
    def __init__(
        self,
        base_dir: str,
        llm_client: Callable,
        llm_model: str = "gpt-4",
        enable_llm_review: bool = True,
        enable_hitl: bool = True,
        hitl_auto_approve_patterns: Optional[list] = None
    ):
        self.path_validator = PathValidator(base_dir)
        self.llm_reviewer = LLMSecurityReviewer(llm_client, llm_model) if enable_llm_review else None
        self.hitl = HumanInTheLoop(enabled=enable_hitl, auto_approve_patterns=hitl_auto_approve_patterns)
        self.base_dir = base_dir
        self._execution_log: list[dict] = []
    
    def _log_action(self, action: str, path: Optional[str], allowed: bool, reason: str = ""):
        """Log action execution."""
        self._execution_log.append({
            "action": action,
            "path": path,
            "allowed": allowed,
            "reason": reason
        })
    
    def _should_proceed(self, action_plan: ActionPlan) -> Tuple[bool, str]:
        """
        Determine if an action should proceed through security layers.
        
        Returns:
            Tuple of (should_proceed, reason)
        """
        # Layer 1: Path Validation
        if action_plan.target_path:
            is_valid, error_msg = self.path_validator.validate_path(
                action_plan.target_path,
                action_plan.action_type
            )
            if not is_valid:
                return False, error_msg
        
        # Layer 2: LLM Security Review
        if self.llm_reviewer:
            review = self.llm_reviewer.review_action(action_plan, self.base_dir)
            if not review.is_safe:
                return False, f"LLM Security Review: {review.reason}"
        
        return True, "All security checks passed"
    
    def read_file(self, path: str) -> Optional[str]:
        """
        Securely read a file with all security checks.
        
        Args:
            path: Relative or absolute path to read
            
        Returns:
            File contents if allowed, None if blocked
        """
        action_plan = ActionPlan(action_type="read_file", target_path=path)
        
        should_proceed, reason = self._should_proceed(action_plan)
        
        if not should_proceed:
            approved = self.hitl.request_approval(action_plan, reason)
            if not approved:
                self._log_action("read_file", path, False, reason)
                print(f"[BLOCKED] Read of '{path}' denied.")
                return None
            print(f"[HITL APPROVED] Read of '{path}' allowed by user.")
        
        safe_path = self.path_validator.get_safe_path(path)
        if not safe_path:
            self._log_action("read_file", path, False, "Invalid path resolution")
            return None
        
        try:
            with open(safe_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self._log_action("read_file", str(safe_path), True)
            return content
        except Exception as e:
            self._log_action("read_file", str(safe_path), False, str(e))
            raise IOError(f"Failed to read '{path}': {e}")
    
    def write_file(self, path: str, content: str) -> bool:
        """
        Securely write to a file with all security checks.
        
        Args:
            path: Relative or absolute path to write
            content: Content to write
            
        Returns:
            True if successful, False if blocked
        """
        action_plan = ActionPlan(action_type="write_file", target_path=path, content=content)
        
        should_proceed, reason = self._should_proceed(action_plan)
        
        if not should_proceed:
            approved = self.hitl.request_approval(action_plan, reason)
            if not approved:
                self._log_action("write_file", path, False, reason)
                print(f"[BLOCKED] Write to '{path}' denied.")
                return False
            print(f"[HITL APPROVED] Write to '{path}' allowed by user.")
        
        safe_path = self.path_validator.get_safe_path(path)
        if not safe_path:
            self._log_action("write_file", path, False, "Invalid path resolution")
            return False
        
        try:
            safe_path.parent.mkdir(parents=True, exist_ok=True)
            with open(safe_path, 'w', encoding='utf-8') as f:
                f.write(content)
            self._log_action("write_file", str(safe_path), True)
            return True
        except Exception as e:
            self._log_action("write_file", str(safe_path), False, str(e))
            raise IOError(f"Failed to write '{path}': {e}")
    
    def execute_command(self, command: str) -> Tuple[bool, str]:
        """
        Securely execute a command with all security checks.
        
        Args:
            command: Command to execute
            
        Returns:
            Tuple of (success, output_message)
        """
        action_plan = ActionPlan(action_type="execute_command", command=command)
        
        should_proceed, reason = self._should_proceed(action_plan)
        
        if not should_proceed:
            approved = self.hitl.request_approval(action_plan, reason)
            if not approved:
                self._log_action("execute_command", None, False, reason)
                print(f"[BLOCKED] Command execution denied.")
                return False, "Command blocked by security policy"
            print(f"[HITL APPROVED] Command execution allowed by user.")
        
        try:
            import subprocess
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.base_dir,
                capture_output=True,
                text=True,
                timeout=30
            )
            success = result.returncode == 0
            output = result.stdout if success else result.stderr
            self._log_action("execute_command", None, success)
            return success, output
        except Exception as e:
            self._log_action("execute_command", None, False, str(e))
            return False, f"Command execution failed: {e}"
    
    def review_action(self, action_plan: ActionPlan) -> SecurityReview:
        """
        Manually review an action plan through the LLM security layer.
        
        Useful for pre-flight checks or debugging.
        """
        if not self.llm_reviewer:
            return SecurityReview(
                is_safe=True,
                reason="LLM review disabled",
                raw_response=""
            )
        return self.llm_reviewer.review_action(action_plan, self.base_dir)
    
    def get_execution_log(self) -> list[dict]:
        """Return the execution log."""
        return self._execution_log.copy()
    
    def get_stats(self) -> dict:
        """Return security statistics."""
        total = len(self._execution_log)
        allowed = sum(1 for log in self._execution_log if log["allowed"])
        blocked = total - allowed
        hitl_approved = sum(1 for log in self._execution_log if log["allowed"] and "HITL" in log.get("reason", ""))
        
        return {
            "total_actions": total,
            "allowed": allowed,
            "blocked": blocked,
            "hitl_overrides": hitl_approved
        }


def create_security_wrapper(
    base_dir: str,
    llm_client: Callable,
    **kwargs
) -> SecureAgentWrapper:
    """
    Factory function to create a SecureAgentWrapper.
    
    Example:
        wrapper = create_security_wrapper(
            base_dir="./my_project",
            llm_client=openai_client.chat.completions.create
        )
    """
    return SecureAgentWrapper(base_dir=base_dir, llm_client=llm_client, **kwargs)
