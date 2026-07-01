"""
Base Agent Interface and Common Utilities
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AgentMessage:
    """Standard message format for inter-agent communication"""

    sender: str
    receiver: str
    msg_type: str  # "request", "response", "error", "info"
    content: Dict[str, Any]
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    message_id: str = field(
        default_factory=lambda: f"msg_{datetime.now(timezone.utc).timestamp()}"
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class ApplicationData:
    """Standard loan application data structure"""

    application_id: str
    applicant_info: Dict[str, Any]
    financial_info: Dict[str, Any]
    documents: List[Dict[str, Any]]
    credit_history: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DecisionResult:
    """Underwriting decision structure"""

    application_id: str
    decision: str  # "approve", "deny", "review"
    confidence: float
    risk_score: float
    recommended_terms: Optional[Dict[str, Any]] = None
    rationale: str = ""
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    agent_outputs: Dict[str, Any] = field(default_factory=dict)
    fairness_metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class BaseAgent(ABC):
    """Abstract base class for all agents"""

    def __init__(self, agent_id: str, config: Optional[Dict[str, Any]] = None):
        self.agent_id = agent_id
        self.config = config or {}
        self.message_history: List[AgentMessage] = []
        self.logger = logging.getLogger(f"{__name__}.{agent_id}")

    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """Main processing method - must be implemented by subclass"""

    def send_message(
        self, receiver: str, msg_type: str, content: Dict[str, Any]
    ) -> AgentMessage:
        """Send a message to another agent"""
        msg = AgentMessage(
            sender=self.agent_id, receiver=receiver, msg_type=msg_type, content=content
        )
        self.message_history.append(msg)
        self.logger.info(f"Sent {msg_type} to {receiver}: {msg.message_id}")
        return msg

    def receive_message(self, message: AgentMessage) -> None:
        """Receive and log a message"""
        self.message_history.append(message)
        self.logger.info(
            f"Received {message.msg_type} from {message.sender}: {message.message_id}"
        )

    def get_status(self) -> Dict[str, Any]:
        """Return agent status"""
        return {
            "agent_id": self.agent_id,
            "messages_processed": len(self.message_history),
            "status": "active",
        }


class AgentRegistry:
    """Registry for managing agent instances"""

    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}

    def register(self, agent: BaseAgent) -> None:
        """Register an agent"""
        self.agents[agent.agent_id] = agent
        logger.info(f"Registered agent: {agent.agent_id}")

    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Retrieve an agent by ID"""
        return self.agents.get(agent_id)

    def get_all_agents(self) -> List[BaseAgent]:
        """Get list of all registered agents"""
        return list(self.agents.values())

    def get_status_report(self) -> Dict[str, Any]:
        """Get status of all agents"""
        return {agent_id: agent.get_status() for agent_id, agent in self.agents.items()}
