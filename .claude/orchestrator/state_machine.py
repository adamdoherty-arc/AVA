"""
LangGraph-Inspired State Machine for Orchestrator
Provides checkpointing, rollback, and state persistence
Version: 2.0
"""
from enum import Enum
from typing import Dict, Any, Optional, List
import json
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class OrchestratorState(Enum):
    """Orchestrator workflow states"""
    IDLE = "idle"
    VALIDATING_REQUEST = "validating_request"
    LOADING_SPECS = "loading_specs"
    SELECTING_AGENTS = "selecting_agents"
    EXECUTING_AGENTS = "executing_agents"
    RUNNING_QA = "running_qa"
    RUNNING_UI_TESTS = "running_ui_tests"
    GENERATING_SUMMARY = "generating_summary"
    COMPLETED = "completed"
    FAILED = "failed"


class StateTransition:
    """Represents a state transition"""
    def __init__(self, from_state: OrchestratorState, to_state: OrchestratorState,
                 context_update: Optional[Dict] = None, timestamp: Optional[datetime] = None):
        self.from_state = from_state
        self.to_state = to_state
        self.context_update = context_update or {}
        self.timestamp = timestamp or datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from": self.from_state.value,
            "to": self.to_state.value,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context_update
        }


class StateMachine:
    """
    State machine for orchestrator workflow
    Supports checkpointing, rollback, and state persistence
    Inspired by LangGraph's state management
    """

    def __init__(self, checkpoint_dir: str = ".claude/orchestrator/checkpoints"):
        self.state = OrchestratorState.IDLE
        self.context: Dict[str, Any] = {}
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.history: List[StateTransition] = []
        self.current_checkpoint_id: Optional[str] = None

    def transition(self, new_state: OrchestratorState, context_update: Optional[Dict] = None) -> bool:
        """
        Transition to new state with checkpoint

        Args:
            new_state: Target state
            context_update: Optional context updates

        Returns:
            True if transition successful
        """
        if not self._is_valid_transition(self.state, new_state):
            logger.warning(f"Invalid transition: {self.state.value} -> {new_state.value}")
            return False

        # Record transition
        transition = StateTransition(self.state, new_state, context_update)
        self.history.append(transition)

        # Update state
        old_state = self.state
        self.state = new_state

        # Update context
        if context_update:
            self.context.update(context_update)

        # Checkpoint
        self._checkpoint()

        logger.info(f"State transition: {old_state.value} -> {new_state.value}")
        return True

    def _is_valid_transition(self, from_state: OrchestratorState,
                            to_state: OrchestratorState) -> bool:
        """Validate if transition is allowed"""

        # Define valid state transitions
        valid_transitions = {
            OrchestratorState.IDLE: [
                OrchestratorState.VALIDATING_REQUEST
            ],
            OrchestratorState.VALIDATING_REQUEST: [
                OrchestratorState.LOADING_SPECS,
                OrchestratorState.FAILED
            ],
            OrchestratorState.LOADING_SPECS: [
                OrchestratorState.SELECTING_AGENTS,
                OrchestratorState.FAILED
            ],
            OrchestratorState.SELECTING_AGENTS: [
                OrchestratorState.EXECUTING_AGENTS,
                OrchestratorState.FAILED
            ],
            OrchestratorState.EXECUTING_AGENTS: [
                OrchestratorState.RUNNING_QA,
                OrchestratorState.FAILED
            ],
            OrchestratorState.RUNNING_QA: [
                OrchestratorState.RUNNING_UI_TESTS,
                OrchestratorState.GENERATING_SUMMARY,
                OrchestratorState.FAILED
            ],
            OrchestratorState.RUNNING_UI_TESTS: [
                OrchestratorState.GENERATING_SUMMARY,
                OrchestratorState.FAILED
            ],
            OrchestratorState.GENERATING_SUMMARY: [
                OrchestratorState.COMPLETED,
                OrchestratorState.FAILED
            ],
            OrchestratorState.COMPLETED: [
                OrchestratorState.IDLE
            ],
            OrchestratorState.FAILED: [
                OrchestratorState.IDLE
            ]
        }

        allowed_states = valid_transitions.get(from_state, [])
        return to_state in allowed_states

    def _checkpoint(self) -> str:
        """
        Save current state to disk

        Returns:
            Checkpoint ID
        """
        checkpoint_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        checkpoint = {
            "checkpoint_id": checkpoint_id,
            "state": self.state.value,
            "context": self.context,
            "history": [t.to_dict() for t in self.history],
            "timestamp": datetime.now().isoformat()
        }

        checkpoint_file = self.checkpoint_dir / f"checkpoint_{checkpoint_id}.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)

        self.current_checkpoint_id = checkpoint_id
        logger.debug(f"Checkpoint created: {checkpoint_id}")

        # Cleanup old checkpoints (keep last 100)
        self._cleanup_old_checkpoints(keep=100)

        return checkpoint_id

    def _cleanup_old_checkpoints(self, keep: int = 100):
        """Remove old checkpoints, keeping only the most recent N"""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.json"))

        if len(checkpoints) > keep:
            for old_checkpoint in checkpoints[:-keep]:
                old_checkpoint.unlink()
                logger.debug(f"Removed old checkpoint: {old_checkpoint.name}")

    def rollback(self, steps: int = 1) -> bool:
        """
        Rollback to previous state

        Args:
            steps: Number of steps to roll back

        Returns:
            True if rollback successful
        """
        if len(self.history) < steps:
            logger.error(f"Cannot rollback {steps} steps, only {len(self.history)} in history")
            return False

        # Remove last N transitions
        for _ in range(steps):
            self.history.pop()

        # Restore state from last transition
        if self.history:
            last_transition = self.history[-1]
            self.state = last_transition.to_state
            logger.info(f"Rolled back {steps} steps to state: {self.state.value}")
        else:
            self.state = OrchestratorState.IDLE
            self.context = {}
            logger.info("Rolled back to initial state")

        # Checkpoint the rollback
        self._checkpoint()
        return True

    def load_checkpoint(self, checkpoint_file: str) -> bool:
        """
        Load state from checkpoint file

        Args:
            checkpoint_file: Path to checkpoint file

        Returns:
            True if loaded successfully
        """
        try:
            checkpoint_path = Path(checkpoint_file)
            if not checkpoint_path.exists():
                checkpoint_path = self.checkpoint_dir / checkpoint_file

            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)

            self.state = OrchestratorState(checkpoint['state'])
            self.context = checkpoint['context']
            self.history = [
                StateTransition(
                    from_state=OrchestratorState(t['from']),
                    to_state=OrchestratorState(t['to']),
                    context_update=t.get('context', {}),
                    timestamp=datetime.fromisoformat(t['timestamp'])
                )
                for t in checkpoint['history']
            ]
            self.current_checkpoint_id = checkpoint['checkpoint_id']

            logger.info(f"Loaded checkpoint: {checkpoint_path.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get the most recent checkpoint file"""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.json"))
        return checkpoints[-1] if checkpoints else None

    def reset(self) -> None:
        """Reset to initial state"""
        self.state = OrchestratorState.IDLE
        self.context = {}
        self.history = []
        self.current_checkpoint_id = None
        logger.info("State machine reset")

    def get_state_summary(self) -> Dict[str, Any]:
        """Get current state summary"""
        return {
            "current_state": self.state.value,
            "context_keys": list(self.context.keys()),
            "history_length": len(self.history),
            "current_checkpoint": self.current_checkpoint_id
        }

    def export_history(self) -> List[Dict[str, Any]]:
        """Export complete state transition history"""
        return [t.to_dict() for t in self.history]


# Singleton instance
_state_machine_instance: Optional[StateMachine] = None


def get_state_machine() -> StateMachine:
    """Get singleton state machine instance"""
    global _state_machine_instance
    if _state_machine_instance is None:
        _state_machine_instance = StateMachine()
    return _state_machine_instance