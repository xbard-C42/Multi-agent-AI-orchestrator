"""
Voice Interface Agent Coordination Pipelines
Multi-modal queuing for human interaction and internal governance
"""

import asyncio
from typing import Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging

class InteractionMode(Enum):
    """Different types of interaction pipelines"""
    HUMAN_CONVERSATION = "human_conversation"      # Direct human-agent dialogue
    INTERNAL_GOVERNANCE = "internal_governance"    # Agent-agent collaboration
    HUMAN_COUNCIL = "human_council"               # Human participating in agent council
    BACKGROUND_PROCESSING = "background_processing" # Silent agent work
    INTERRUPT_HANDLING = "interrupt_handling"      # Emergency/urgent interventions

class VoiceContext(Enum):
    """Voice interaction contexts that affect agent selection"""
    CASUAL_QUERY = "casual_query"
    TECHNICAL_EXPLANATION = "technical_explanation"
    CREATIVE_COLLABORATION = "creative_collaboration"
    PROBLEM_SOLVING = "problem_solving"
    EMOTIONAL_SUPPORT = "emotional_support"
    LEARNING_SESSION = "learning_session"
    DECISION_CONSULTATION = "decision_consultation"

class SpeechPriority(Enum):
    """Priority levels for voice interactions"""
    EMERGENCY = 1          # Safety/security alerts
    URGENT_RESPONSE = 2    # User waiting for immediate answer
    CONTEXTUAL_INPUT = 3   # Relevant expertise for current topic
    BACKGROUND_INFO = 4    # Additional information/context
    SOCIAL_ENGAGEMENT = 5  # Relationship building/casual chat

@dataclass
class VoiceInteractionRequest:
    """Request for an agent to speak to human"""
    agent_id: str
    content_preview: str
    interaction_mode: InteractionMode
    voice_context: VoiceContext
    priority: SpeechPriority
    estimated_duration: float  # seconds
    requires_human_response: bool
    interrupt_allowed: bool = True
    context_data: Dict = field(default_factory=dict)

@dataclass
class PipelineState:
    """Current state of an interaction pipeline"""
    pipeline_id: str
    mode: InteractionMode
    active_agent: Optional[str] = None
    queued_requests: List[VoiceInteractionRequest] = field(default_factory=list)
    context_stack: List[VoiceContext] = field(default_factory=list)
    human_attention_state: str = "available"  # available, busy, away
    last_interaction: Optional[datetime] = None

class VoiceCoordinationEngine:
    """Coordinates voice interactions across multiple agent pipelines"""
    
    def __init__(self, governance_orchestrator):
        self.governance_orchestrator = governance_orchestrator
        self.pipelines: Dict[str, PipelineState] = {}
        self.agent_voice_profiles: Dict[str, 'VoiceProfile'] = {}
        self.human_context_tracker = HumanContextTracker()
        self.interrupt_manager = InterruptManager()
        
        # Initialize default pipelines
        self._initialize_default_pipelines()
    
    def _initialize_default_pipelines(self):
        """Create default interaction pipelines"""
        pipeline_configs = [
            ("primary_conversation", InteractionMode.HUMAN_CONVERSATION),
            ("governance_council", InteractionMode.INTERNAL_GOVERNANCE),
            ("background_tasks", InteractionMode.BACKGROUND_PROCESSING),
            ("emergency_channel", InteractionMode.INTERRUPT_HANDLING)
        ]
        
        for pipeline_id, mode in pipeline_configs:
            self.pipelines[pipeline_id] = PipelineState(pipeline_id, mode)
    
    async def request_speech_turn(self, request: VoiceInteractionRequest) -> Dict:
        """Agent requests permission to speak to human"""
        
        # Step 1: Determine appropriate pipeline
        pipeline_id = self._select_pipeline(request)
        pipeline = self.pipelines[pipeline_id]
        
        # Step 2: Calculate speech priority score
        priority_score = await self._calculate_speech_priority(request, pipeline)
        
        # Step 3: Check for immediate speech or queue
        if await self._can_speak_immediately(request, pipeline, priority_score):
            return await self._grant_immediate_speech(request, pipeline)
        else:
            return await self._queue_for_later_speech(request, pipeline, priority_score)
    
    async def _calculate_speech_priority(self, request: VoiceInteractionRequest, 
                                       pipeline: PipelineState) -> float:
        """Calculate priority score for speech request"""
        
        base_priority = {
            SpeechPriority.EMERGENCY: 1.0,
            SpeechPriority.URGENT_RESPONSE: 0.8,
            SpeechPriority.CONTEXTUAL_INPUT: 0.6,
            SpeechPriority.BACKGROUND_INFO: 0.4,
            SpeechPriority.SOCIAL_ENGAGEMENT: 0.2
        }[request.priority]
        
        # Expertise relevance multiplier
        expertise_bonus = await self._calculate_expertise_relevance(
            request.agent_id, request.voice_context, request.content_preview
        )
        
        # Human attention context multiplier
        attention_multiplier = self.human_context_tracker.get_attention_multiplier(
            request.voice_context
        )
        
        # Conversation flow bonus/penalty
        flow_adjustment = await self._calculate_conversation_flow_score(
            request, pipeline.context_stack
        )
        
        # Agent voice quality factor
        voice_quality = self.agent_voice_profiles.get(request.agent_id, VoiceProfile()).quality_score
        
        final_score = (
            base_priority * 0.4 +
            expertise_bonus * 0.3 +
            flow_adjustment * 0.2 +
            voice_quality * 0.1
        ) * attention_multiplier
        
        return min(1.0, final_score)
    
    async def _can_speak_immediately(self, request: VoiceInteractionRequest,
                                   pipeline: PipelineState, priority_score: float) -> bool:
        """Determine if agent can speak immediately"""
        
        # Emergency always gets immediate access
        if request.priority == SpeechPriority.EMERGENCY:
            return True
        
        # Check if pipeline is available
        if pipeline.active_agent is None:
            return True
        
        # Check if this request should interrupt current speaker
        if request.interrupt_allowed and priority_score > 0.8:
            current_priority = await self._get_current_speaker_priority(pipeline)
            return priority_score > current_priority + 0.2  # Threshold to prevent constant interruption
        
        return False
    
    async def _grant_immediate_speech(self, request: VoiceInteractionRequest,
                                    pipeline: PipelineState) -> Dict:
        """Grant immediate speech permission to agent"""
        
        # Handle interruption if necessary
        if pipeline.active_agent:
            await self._handle_interruption(pipeline, request)
        
        # Grant speech
        pipeline.active_agent = request.agent_id
        pipeline.context_stack.append(request.voice_context)
        pipeline.last_interaction = datetime.now()
        
        # Log for coordination with governance system
        await self._log_voice_interaction(request, "immediate_grant")
        
        return {
            'status': 'granted',
            'speech_window_seconds': self._calculate_speech_window(request),
            'pipeline': pipeline.pipeline_id,
            'interrupt_allowed': request.interrupt_allowed
        }
    
    async def _queue_for_later_speech(self, request: VoiceInteractionRequest,
                                    pipeline: PipelineState, priority_score: float) -> Dict:
        """Queue speech request for later execution"""
        
        # Insert into priority queue
        inserted = False
        for i, queued_request in enumerate(pipeline.queued_requests):
            queued_priority = await self._calculate_speech_priority(queued_request, pipeline)
            if priority_score > queued_priority:
                pipeline.queued_requests.insert(i, request)
                inserted = True
                break
        
        if not inserted:
            pipeline.queued_requests.append(request)
        
        # Estimate wait time
        estimated_wait = await self._estimate_queue_wait_time(pipeline, request)
        
        return {
            'status': 'queued',
            'position_in_queue': pipeline.queued_requests.index(request) + 1,
            'estimated_wait_seconds': estimated_wait,
            'pipeline': pipeline.pipeline_id
        }
    
    async def release_speech_turn(self, agent_id: str, pipeline_id: str) -> Dict:
        """Agent releases speech turn, potentially triggering next speaker"""
        
        pipeline = self.pipelines[pipeline_id]
        
        if pipeline.active_agent == agent_id:
            pipeline.active_agent = None
            
            # Process next in queue
            next_speaker = await self._get_next_speaker(pipeline)
            if next_speaker:
                return await self._grant_immediate_speech(next_speaker, pipeline)
        
        return {'status': 'released', 'next_speaker': None}
    
    def _select_pipeline(self, request: VoiceInteractionRequest) -> str:
        """Select appropriate pipeline for request"""
        
        mode_to_pipeline = {
            InteractionMode.HUMAN_CONVERSATION: "primary_conversation",
            InteractionMode.INTERNAL_GOVERNANCE: "governance_council", 
            InteractionMode.BACKGROUND_PROCESSING: "background_tasks",
            InteractionMode.INTERRUPT_HANDLING: "emergency_channel"
        }
        
        return mode_to_pipeline.get(request.interaction_mode, "primary_conversation")
    
    async def _calculate_expertise_relevance(self, agent_id: str, context: VoiceContext, 
                                           content: str) -> float:
        """Calculate how relevant agent's expertise is for voice context"""
        
        # Use existing ranking engine from governance system
        if hasattr(self.governance_orchestrator, 'ranking_engine'):
            # Convert voice context to domain context
            domain_mapping = {
                VoiceContext.TECHNICAL_EXPLANATION: "technical",
                VoiceContext.CREATIVE_COLLABORATION: "creative",
                VoiceContext.PROBLEM_SOLVING: "analytical",
                VoiceContext.EMOTIONAL_SUPPORT: "social"
            }
            
            domain = domain_mapping.get(context, "general")
            return self.governance_orchestrator.ranking_engine.calculate_agent_score(
                agent_id, domain, content, []
            )
        
        return 0.5  # Default relevance
    
    async def _calculate_conversation_flow_score(self, request: VoiceInteractionRequest,
                                               context_stack: List[VoiceContext]) -> float:
        """Calculate bonus/penalty based on conversation flow"""
        
        if not context_stack:
            return 0.0
        
        current_context = context_stack[-1]
        
        # Bonus for maintaining context coherence
        if request.voice_context == current_context:
            return 0.2
        
        # Penalty for jarring context switches
        context_compatibility = {
            VoiceContext.TECHNICAL_EXPLANATION: [VoiceContext.PROBLEM_SOLVING],
            VoiceContext.CREATIVE_COLLABORATION: [VoiceContext.PROBLEM_SOLVING],
            VoiceContext.CASUAL_QUERY: [VoiceContext.SOCIAL_ENGAGEMENT]
        }
        
        if request.voice_context in context_compatibility.get(current_context, []):
            return 0.1
        
        # Large penalty for incompatible switches
        incompatible_switches = [
            (VoiceContext.EMOTIONAL_SUPPORT, VoiceContext.TECHNICAL_EXPLANATION),
            (VoiceContext.CASUAL_QUERY, VoiceContext.DECISION_CONSULTATION)
        ]
        
        if (current_context, request.voice_context) in incompatible_switches:
            return -0.3
        
        return 0.0  # Neutral for other transitions
    
    async def _handle_interruption(self, pipeline: PipelineState, 
                                 interrupting_request: VoiceInteractionRequest):
        """Handle interruption of current speaker"""
        
        current_agent = pipeline.active_agent
        
        # Log interruption for performance tracking
        await self._log_voice_interaction(interrupting_request, "interruption", {
            'interrupted_agent': current_agent,
            'interruption_reason': interrupting_request.priority.value
        })
        
        # If current agent can be paused, add back to queue
        if current_agent and self._agent_supports_pause(current_agent):
            pause_request = VoiceInteractionRequest(
                agent_id=current_agent,
                content_preview="[RESUMING PREVIOUS STATEMENT]",
                interaction_mode=InteractionMode.HUMAN_CONVERSATION,
                voice_context=pipeline.context_stack[-1] if pipeline.context_stack else VoiceContext.CASUAL_QUERY,
                priority=SpeechPriority.CONTEXTUAL_INPUT,
                estimated_duration=5.0,
                requires_human_response=False
            )
            pipeline.queued_requests.insert(0, pause_request)  # High priority for resumption
    
    def _calculate_speech_window(self, request: VoiceInteractionRequest) -> float:
        """Calculate how long agent should be allowed to speak"""
        
        base_duration = request.estimated_duration
        
        # Adjust based on priority
        priority_multipliers = {
            SpeechPriority.EMERGENCY: 2.0,
            SpeechPriority.URGENT_RESPONSE: 1.5,
            SpeechPriority.CONTEXTUAL_INPUT: 1.0,
            SpeechPriority.BACKGROUND_INFO: 0.8,
            SpeechPriority.SOCIAL_ENGAGEMENT: 1.2
        }
        
        multiplier = priority_multipliers.get(request.priority, 1.0)
        
        # Cap duration to prevent monopolization
        max_duration = 60.0 if request.priority in [SpeechPriority.EMERGENCY, SpeechPriority.URGENT_RESPONSE] else 30.0
        
        return min(base_duration * multiplier, max_duration)
    
    async def _estimate_queue_wait_time(self, pipeline: PipelineState, 
                                      request: VoiceInteractionRequest) -> float:
        """Estimate wait time in queue"""
        
        total_wait = 0.0
        
        # Current speaker remaining time
        if pipeline.active_agent:
            total_wait += 10.0  # Average remaining time estimate
        
        # Queue processing time
        request_position = pipeline.queued_requests.index(request)
        for i in range(request_position):
            queued_request = pipeline.queued_requests[i]
            total_wait += self._calculate_speech_window(queued_request)
        
        return total_wait
    
    async def _get_next_speaker(self, pipeline: PipelineState) -> Optional[VoiceInteractionRequest]:
        """Get next speaker from queue"""
        
        if pipeline.queued_requests:
            return pipeline.queued_requests.pop(0)
        return None
    
    def _agent_supports_pause(self, agent_id: str) -> bool:
        """Check if agent supports pause/resume functionality"""
        voice_profile = self.agent_voice_profiles.get(agent_id)
        return voice_profile.supports_interruption if voice_profile else False
    
    async def _log_voice_interaction(self, request: VoiceInteractionRequest, 
                                   event_type: str, metadata: Dict = None):
        """Log voice interaction for performance analysis"""
        
        log_entry = {
            'timestamp': datetime.now(),
            'agent_id': request.agent_id,
            'event_type': event_type,
            'interaction_mode': request.interaction_mode.value,
            'voice_context': request.voice_context.value,
            'priority': request.priority.value,
            'metadata': metadata or {}
        }
        
        # This would integrate with the governance system's logging
        if hasattr(self.governance_orchestrator, 'failure_prevention_log'):
            self.governance_orchestrator.failure_prevention_log.append(log_entry)


@dataclass
class VoiceProfile:
    """Voice interaction profile for an agent"""
    quality_score: float = 0.7
    supports_interruption: bool = True
    preferred_contexts: List[VoiceContext] = field(default_factory=list)
    average_response_time: float = 2.0
    speech_patterns: Dict = field(default_factory=dict)


class HumanContextTracker:
    """Tracks human's attention and context for better agent coordination"""
    
    def __init__(self):
        self.current_activity = "available"
        self.attention_level = 1.0
        self.conversation_history = []
        self.preferences = {}
    
    def get_attention_multiplier(self, context: VoiceContext) -> float:
        """Get multiplier based on human's current attention state"""
        
        base_multiplier = self.attention_level
        
        # Context-specific adjustments
        if self.current_activity == "busy":
            if context in [VoiceContext.EMERGENCY, VoiceContext.URGENT_RESPONSE]:
                return base_multiplier * 1.5  # Important things get through
            else:
                return base_multiplier * 0.3  # Less important things wait
        
        return base_multiplier


class InterruptManager:
    """Manages interruption logic and prevents chaos"""
    
    def __init__(self):
        self.recent_interruptions = []
        self.interruption_cooldown = 5.0  # seconds
    
    def can_interrupt(self, agent_id: str, priority: SpeechPriority) -> bool:
        """Check if agent can interrupt current speaker"""
        
        # Emergency always allowed
        if priority == SpeechPriority.EMERGENCY:
            return True
        
        # Check cooldown period
        now = datetime.now()
        recent = [
            t for t in self.recent_interruptions 
            if (now - t).total_seconds() < self.interruption_cooldown
        ]
        
        # Too many recent interruptions
        if len(recent) >= 2:
            return False
        
        return True


# Integration with governance scenarios
class VoiceGovernanceIntegration:
    """Shows how voice coordination integrates with governance decisions"""
    
    @staticmethod
    async def scenario_human_joins_agent_council():
        """
        Human joins ongoing agent governance discussion
        Voice system must coordinate between:
        - Internal agent-agent governance queue
        - Human-facing explanation queue  
        - Decision consultation queue
        """
        
        print("=== SCENARIO: Human Joins Agent Council ===")
        
        # Governance discussion happening
        print("Background: Agents discussing database migration strategy")
        print("Event: Human joins conversation")
        
        # Voice system must now manage multiple pipelines:
        print("\nPipeline Coordination:")
        print("1. GOVERNANCE: Continue agent-agent technical discussion")
        print("2. HUMAN_BRIEFING: Explain current state to human")  
        print("3. CONSULTATION: Get human input on key decisions")
        print("4. INTEGRATION: Incorporate human feedback into governance")
        
        # Different agents optimal for different pipelines
        print("\nAgent Selection by Pipeline:")
        print("- Technical governance: Database expert continues")
        print("- Human briefing: Communication specialist explains")
        print("- Decision consultation: Facilitator gathers input")
        print("- Integration: Synthesizer merges all perspectives")
        
        print("✅ Multi-pipeline coordination prevents chaos")
    
    @staticmethod
    async def scenario_emergency_interrupt_governance():
        """
        Emergency interrupt during governance session
        Must coordinate immediate human notification + governance pause
        """
        
        print("\n=== SCENARIO: Emergency Interrupt ===")
        
        print("Background: Agent council debating API versioning")
        print("Event: Security breach detected")
        
        print("\nCoordination Response:")
        print("1. INTERRUPT: Security agent gets immediate voice access")
        print("2. PAUSE: Governance discussion state saved")
        print("3. ESCALATE: Human immediately notified")
        print("4. CONTEXT_SWITCH: All agents shift to security mode")
        print("5. RESUME: Return to governance with new security constraints")
        
        print("✅ Emergency handling preserves both human safety and governance continuity")


# Demo the complete voice coordination system
async def demo_voice_coordination():
    """Demonstrate voice interface coordination with governance"""
    
    print("🎙️  VOICE INTERFACE COORDINATION DEMO")
    print("="*60)
    
    # Mock governance orchestrator
    class MockGovernanceOrchestrator:
        def __init__(self):
            self.failure_prevention_log = []
    
    governance = MockGovernanceOrchestrator()
    voice_engine = VoiceCoordinationEngine(governance)
    
    # Register agent voice profiles
    voice_engine.agent_voice_profiles.update({
        "technical_expert": VoiceProfile(
            quality_score=0.9,
            preferred_contexts=[VoiceContext.TECHNICAL_EXPLANATION, VoiceContext.PROBLEM_SOLVING]
        ),
        "communication_specialist": VoiceProfile(
            quality_score=0.95,
            preferred_contexts=[VoiceContext.CASUAL_QUERY, VoiceContext.LEARNING_SESSION]
        ),
        "security_agent": VoiceProfile(
            quality_score=0.8,
            preferred_contexts=[VoiceContext.DECISION_CONSULTATION]
        )
    })
    
    # Simulate multiple speech requests
    requests = [
        VoiceInteractionRequest(
            agent_id="communication_specialist",
            content_preview="Hello! I can help explain what we're working on",
            interaction_mode=InteractionMode.HUMAN_CONVERSATION,
            voice_context=VoiceContext.CASUAL_QUERY,
            priority=SpeechPriority.SOCIAL_ENGAGEMENT,
            estimated_duration=8.0,
            requires_human_response=True
        ),
        VoiceInteractionRequest(
            agent_id="technical_expert", 
            content_preview="The database migration has three key considerations",
            interaction_mode=InteractionMode.HUMAN_CONVERSATION,
            voice_context=VoiceContext.TECHNICAL_EXPLANATION,
            priority=SpeechPriority.CONTEXTUAL_INPUT,
            estimated_duration=15.0,
            requires_human_response=False
        ),
        VoiceInteractionRequest(
            agent_id="security_agent",
            content_preview="SECURITY ALERT: Unusual access pattern detected",
            interaction_mode=InteractionMode.INTERRUPT_HANDLING,
            voice_context=VoiceContext.DECISION_CONSULTATION,
            priority=SpeechPriority.EMERGENCY,
            estimated_duration=5.0,
            requires_human_response=True,
            interrupt_allowed=True
        )
    ]
    
    print("Processing speech requests...")
    
    results = []
    for request in requests:
        result = await voice_engine.request_speech_turn(request)
        results.append((request.agent_id, result))
        print(f"{request.agent_id}: {result['status']} - Priority: {request.priority.value}")
    
    print(f"\nFinal Pipeline States:")
    for pipeline_id, pipeline in voice_engine.pipelines.items():
        print(f"{pipeline_id}: Active={pipeline.active_agent}, Queued={len(pipeline.queued_requests)}")
    
    await VoiceGovernanceIntegration.scenario_human_joins_agent_council()
    await VoiceGovernanceIntegration.scenario_emergency_interrupt_governance()
    
    print("\n🎯 VOICE COORDINATION BENEFITS:")
    print("✅ Multiple simultaneous interaction pipelines")
    print("✅ Context-aware agent selection for voice interface")
    print("✅ Intelligent interruption handling")
    print("✅ Seamless integration with governance queuing")
    print("✅ Human attention and context tracking")
    print("✅ Emergency response coordination")


if __name__ == "__main__":
    asyncio.run(demo_voice_coordination())
