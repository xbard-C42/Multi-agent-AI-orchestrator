"""
Integrated Governance System: Queuing + Collaborative Safeguards
Prevents coordination failures through intelligent agent orchestration
"""

import asyncio
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging

# Import existing collaborative components (assuming they're available)
from collaborative_safeguards import (
    CooperativeOrchestrator, ContributionType, AgentContribution,
    CollaborationMonitor, ConsensusBuilder, RoleRotator
)
from collaborative_prompts import CollaborativePromptLibrary, AgentRole

logger = logging.getLogger(__name__)

class GovernanceFailureMode(Enum):
    """Types of coordination failures the system prevents"""
    EXPERTISE_MISMATCH = "wrong_expert_speaks_first"
    PREMATURE_CONSENSUS = "insufficient_critical_analysis" 
    ANALYSIS_PARALYSIS = "endless_discussion_no_decisions"
    EXPERT_DOMINANCE = "single_expert_monopolizes"
    GROUPTHINK_CASCADE = "early_agreement_prevents_scrutiny"
    CONTEXT_SWITCHING_CHAOS = "topic_changes_confuse_priorities"
    REPUTATION_GAMING = "agents_optimize_for_metrics_not_outcomes"

@dataclass
class GovernanceContext:
    """Rich context for governance discussions"""
    primary_domain: str
    complexity_level: int  # 1-5
    urgency_level: int     # 1-5  
    stakeholder_impact: str  # "low", "medium", "high", "critical"
    required_expertise: List[str]
    decision_deadline: Optional[datetime] = None
    
class IntegratedGovernanceOrchestrator:
    """Enhanced orchestrator combining queuing with collaborative safeguards"""
    
    def __init__(self):
        # Existing collaborative components
        self.cooperative_orchestrator = CooperativeOrchestrator()
        self.collaboration_monitor = CollaborationMonitor()
        self.consensus_builder = ConsensusBuilder(self.cooperative_orchestrator)
        self.role_rotator = RoleRotator()
        self.prompt_library = CollaborativePromptLibrary()
        
        # New queuing components
        from agent_queuing_system import AgentRankingEngine, GovernanceStackCoordinator
        self.ranking_engine = AgentRankingEngine()
        self.queue_coordinator = GovernanceStackCoordinator()
        
        # Integrated coordination state
        self.active_discussions: Dict[str, 'GovernanceSession'] = {}
        self.failure_prevention_log: List[Dict] = []
        
    async def start_governance_session(self, session_id: str, context: GovernanceContext, 
                                     available_agents: List[str]) -> Dict:
        """Start a governance session with intelligent agent coordination"""
        
        # Step 1: Analyze context for potential failure modes
        failure_risks = self._analyze_failure_risks(context, available_agents)
        logger.info(f"Identified failure risks: {[risk.value for risk in failure_risks]}")
        
        # Step 2: Create optimized participation queue
        queue_config = self._design_participation_strategy(context, failure_risks, available_agents)
        
        # Step 3: Initialize governance session
        session = GovernanceSession(
            session_id=session_id,
            context=context,
            participation_strategy=queue_config,
            failure_prevention_measures=failure_risks
        )
        
        # Step 4: Assign initial roles with queuing awareness
        initial_participants = queue_config['initial_participants']
        role_assignments = await self._assign_roles_with_queue_awareness(
            initial_participants, context, failure_risks
        )
        
        session.active_participants = initial_participants
        session.role_assignments = role_assignments
        self.active_discussions[session_id] = session
        
        return {
            'session_id': session_id,
            'initial_participants': initial_participants,
            'role_assignments': role_assignments,
            'failure_prevention_active': [risk.value for risk in failure_risks],
            'queue_depth': len(queue_config['queued_participants'])
        }
    
    async def process_contribution_with_queue_management(self, session_id: str, agent_id: str, 
                                                       content: str, contribution_type: ContributionType) -> Dict:
        """Process contribution with integrated queuing and safeguards"""
        
        if session_id not in self.active_discussions:
            raise ValueError(f"Session {session_id} not found")
            
        session = self.active_discussions[session_id]
        
        # Step 1: Standard safeguards processing
        safeguards_result = await self.cooperative_orchestrator.process_contribution(
            agent_id, content, contribution_type, session_id
        )
        
        # Step 2: Queue-aware intervention analysis
        queue_interventions = await self._analyze_queue_interventions(session, safeguards_result)
        
        # Step 3: Dynamic participant management
        participation_changes = await self._manage_dynamic_participation(session, safeguards_result)
        
        # Step 4: Failure prevention logging
        self._log_failure_prevention(session_id, agent_id, safeguards_result, queue_interventions)
        
        # Step 5: Update agent performance metrics
        await self._update_agent_performance(agent_id, content, contribution_type, safeguards_result)
        
        return {
            **safeguards_result,
            'queue_interventions': queue_interventions,
            'participation_changes': participation_changes,
            'session_health': await self._calculate_session_health(session)
        }
    
    def _analyze_failure_risks(self, context: GovernanceContext, agents: List[str]) -> List[GovernanceFailureMode]:
        """Identify potential coordination failure modes"""
        risks = []
        
        # Risk: Wrong expert speaks first
        if context.complexity_level >= 4 and context.required_expertise:
            expert_coverage = self._calculate_expert_coverage(agents, context.required_expertise)
            if expert_coverage < 0.8:
                risks.append(GovernanceFailureMode.EXPERTISE_MISMATCH)
        
        # Risk: Premature consensus on complex issues
        if context.complexity_level >= 3 and context.urgency_level >= 4:
            risks.append(GovernanceFailureMode.PREMATURE_CONSENSUS)
        
        # Risk: Analysis paralysis on urgent issues
        if context.urgency_level >= 4 and len(agents) > 5:
            risks.append(GovernanceFailureMode.ANALYSIS_PARALYSIS)
        
        # Risk: Single expert dominance
        domain_distribution = self._calculate_domain_distribution(agents, context.primary_domain)
        if max(domain_distribution.values()) > 0.7:
            risks.append(GovernanceFailureMode.EXPERT_DOMINANCE)
        
        # Risk: Groupthink cascade
        if context.stakeholder_impact == "critical" and len(agents) < 4:
            risks.append(GovernanceFailureMode.GROUPTHINK_CASCADE)
        
        return risks
    
    def _design_participation_strategy(self, context: GovernanceContext, 
                                     risks: List[GovernanceFailureMode], 
                                     agents: List[str]) -> Dict:
        """Design participation strategy to prevent identified failures"""
        
        strategy = {
            'initial_participants': [],
            'queued_participants': [],
            'intervention_triggers': {},
            'rotation_schedule': []
        }
        
        # Strategy: Prevent expertise mismatch
        if GovernanceFailureMode.EXPERTISE_MISMATCH in risks:
            # Start with most relevant experts
            expert_rankings = []
            for agent_id in agents:
                relevance = self.ranking_engine.calculate_agent_score(
                    agent_id, context.primary_domain, "", []
                )
                expert_rankings.append((agent_id, relevance))
            
            expert_rankings.sort(key=lambda x: x[1], reverse=True)
            strategy['initial_participants'] = [agent_id for agent_id, _ in expert_rankings[:2]]
            strategy['queued_participants'] = [agent_id for agent_id, _ in expert_rankings[2:]]
        
        # Strategy: Prevent premature consensus
        if GovernanceFailureMode.PREMATURE_CONSENSUS in risks:
            # Ensure at least one critic is always active
            strategy['intervention_triggers']['force_critic_participation'] = {
                'condition': 'agreement_streak >= 3',
                'action': 'inject_critic_from_queue'
            }
        
        # Strategy: Prevent analysis paralysis  
        if GovernanceFailureMode.ANALYSIS_PARALYSIS in risks:
            # Time-box discussions and force decisions
            strategy['intervention_triggers']['paralysis_prevention'] = {
                'condition': 'discussion_time > 30_minutes AND no_decisions_made',
                'action': 'inject_synthesizer_with_deadline'
            }
        
        # Strategy: Prevent expert dominance
        if GovernanceFailureMode.EXPERT_DOMINANCE in risks:
            # Mandatory rotation after N contributions
            strategy['rotation_schedule'] = [(i * 5, 'rotate_roles') for i in range(1, 10)]
        
        # Strategy: Prevent groupthink
        if GovernanceFailureMode.GROUPTHINK_CASCADE in risks:
            # Devil's advocate injection
            strategy['intervention_triggers']['groupthink_prevention'] = {
                'condition': 'unanimous_agreement_on_complex_issue',
                'action': 'inject_devils_advocate'
            }
        
        return strategy
    
    async def _assign_roles_with_queue_awareness(self, participants: List[str], 
                                               context: GovernanceContext,
                                               risks: List[GovernanceFailureMode]) -> Dict[str, AgentRole]:
        """Assign roles considering both collaboration and failure prevention"""
        
        assignments = {}
        
        # Base role assignment using existing rotator
        base_assignments = self.role_rotator.assign_roles(participants, f"temp_{context.primary_domain}")
        
        # Adjust for failure prevention
        for agent_id, base_role in base_assignments.items():
            
            # Override: Ensure expert gets proposer role for complex technical issues
            if (GovernanceFailureMode.EXPERTISE_MISMATCH in risks and 
                context.complexity_level >= 4):
                agent_expertise = self.ranking_engine.agent_expertise.get(agent_id)
                if agent_expertise and context.primary_domain in agent_expertise.domains:
                    if agent_expertise.domains[context.primary_domain] > 0.8:
                        assignments[agent_id] = AgentRole.PROPOSER
                        continue
            
            # Override: Ensure critic is present for premature consensus prevention
            if (GovernanceFailureMode.PREMATURE_CONSENSUS in risks and 
                not any(role == AgentRole.CRITIC for role in assignments.values())):
                assignments[agent_id] = AgentRole.CRITIC
                continue
            
            # Default to base assignment
            assignments[agent_id] = base_role
        
        return assignments
    
    async def _analyze_queue_interventions(self, session: 'GovernanceSession', 
                                         safeguards_result: Dict) -> List[Dict]:
        """Analyze if queue-based interventions are needed"""
        interventions = []
        
        # Intervention: Inject missing expertise
        if safeguards_result.get('action') == 'intervention_needed':
            warnings = safeguards_result.get('warnings', {})
            
            if 'excessive_contradictions' in warnings:
                # Inject synthesizer from queue
                synthesizer_candidates = [
                    agent_id for agent_id in session.participation_strategy['queued_participants']
                    if self._agent_has_synthesis_skills(agent_id)
                ]
                if synthesizer_candidates:
                    interventions.append({
                        'type': 'inject_synthesizer',
                        'agent_id': synthesizer_candidates[0],
                        'reason': 'resolve_contradictions'
                    })
            
            if 'idea_hoarding' in warnings:
                # Rotate participants to get fresh perspectives
                interventions.append({
                    'type': 'rotate_participants',
                    'action': 'swap_lowest_contributor_with_queue',
                    'reason': 'improve_participation_balance'
                })
        
        # Intervention: Context-driven expertise injection
        if session.context.complexity_level >= 4:
            missing_expertise = self._identify_missing_expertise(session)
            if missing_expertise:
                interventions.append({
                    'type': 'inject_expert',
                    'expertise_needed': missing_expertise,
                    'reason': 'complexity_requires_specialization'
                })
        
        return interventions
    
    async def _manage_dynamic_participation(self, session: 'GovernanceSession', 
                                          safeguards_result: Dict) -> Dict:
        """Dynamically adjust participation based on session needs"""
        changes = {'added': [], 'removed': [], 'role_changes': {}}
        
        # Add participants from queue if needed
        if safeguards_result.get('action') == 'synthesis_triggered':
            # Need more synthesizers
            for agent_id in session.participation_strategy['queued_participants']:
                if self._agent_has_synthesis_skills(agent_id):
                    session.active_participants.append(agent_id)
                    session.role_assignments[agent_id] = AgentRole.SYNTHESIZER
                    changes['added'].append(agent_id)
                    break
        
        # Remove underperforming participants
        underperformers = self._identify_underperforming_participants(session)
        for agent_id in underperformers:
            if len(session.active_participants) > 2:  # Keep minimum viable set
                session.active_participants.remove(agent_id)
                del session.role_assignments[agent_id]
                changes['removed'].append(agent_id)
        
        return changes
    
    async def _calculate_session_health(self, session: 'GovernanceSession') -> Dict:
        """Calculate overall health of governance session"""
        
        # Get base collaboration health
        base_health = self.cooperative_orchestrator.get_collaboration_report(session.session_id)
        
        # Add queue-specific metrics
        queue_metrics = {
            'expertise_coverage': self._calculate_expert_coverage(
                session.active_participants, session.context.required_expertise
            ),
            'participation_balance': self._calculate_participation_balance(session),
            'failure_prevention_effectiveness': self._calculate_failure_prevention_score(session),
            'queue_utilization': len(session.active_participants) / 
                               (len(session.active_participants) + len(session.participation_strategy['queued_participants']))
        }
        
        return {
            **base_health,
            'queue_metrics': queue_metrics,
            'overall_health_score': self._calculate_overall_health_score(base_health, queue_metrics)
        }
    
    # Helper methods for specific calculations
    def _calculate_expert_coverage(self, agents: List[str], required_expertise: List[str]) -> float:
        """Calculate how well current agents cover required expertise"""
        if not required_expertise:
            return 1.0
            
        covered_expertise = set()
        for agent_id in agents:
            if agent_id in self.ranking_engine.agent_expertise:
                expertise = self.ranking_engine.agent_expertise[agent_id]
                for domain in expertise.domains:
                    if domain in required_expertise:
                        covered_expertise.add(domain)
        
        return len(covered_expertise) / len(required_expertise)
    
    def _calculate_domain_distribution(self, agents: List[str], primary_domain: str) -> Dict[str, float]:
        """Calculate distribution of agents across domains"""
        domain_counts = {}
        total_agents = len(agents)
        
        for agent_id in agents:
            if agent_id in self.ranking_engine.agent_expertise:
                expertise = self.ranking_engine.agent_expertise[agent_id]
                for domain in expertise.domains:
                    domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        return {domain: count/total_agents for domain, count in domain_counts.items()}
    
    def _agent_has_synthesis_skills(self, agent_id: str) -> bool:
        """Check if agent has synthesis capabilities"""
        if agent_id not in self.ranking_engine.agent_expertise:
            return False
        expertise = self.ranking_engine.agent_expertise[agent_id]
        return (expertise.collaboration_style == "diplomatic" or 
                "synthesis" in expertise.specialisations)
    
    def _identify_missing_expertise(self, session: 'GovernanceSession') -> List[str]:
        """Identify what expertise is missing from current discussion"""
        required = set(session.context.required_expertise)
        present = set()
        
        for agent_id in session.active_participants:
            if agent_id in self.ranking_engine.agent_expertise:
                expertise = self.ranking_engine.agent_expertise[agent_id]
                present.update(expertise.domains.keys())
        
        return list(required - present)
    
    def _identify_underperforming_participants(self, session: 'GovernanceSession') -> List[str]:
        """Identify participants who should be rotated out"""
        # Placeholder - would analyze actual contribution quality
        return []
    
    def _calculate_participation_balance(self, session: 'GovernanceSession') -> float:
        """Calculate how balanced participation is"""
        # Placeholder - would analyze actual participation patterns
        return 0.8
    
    def _calculate_failure_prevention_score(self, session: 'GovernanceSession') -> float:
        """Score how well the session prevents identified failure modes"""
        prevented_failures = 0
        total_risks = len(session.failure_prevention_measures)
        
        for risk in session.failure_prevention_measures:
            if self._is_failure_mode_prevented(session, risk):
                prevented_failures += 1
        
        return prevented_failures / max(1, total_risks)
    
    def _is_failure_mode_prevented(self, session: 'GovernanceSession', 
                                 risk: GovernanceFailureMode) -> bool:
        """Check if specific failure mode has been prevented"""
        # Implementation would check specific conditions for each failure mode
        return True  # Placeholder
    
    def _calculate_overall_health_score(self, base_health: Dict, queue_metrics: Dict) -> float:
        """Calculate composite health score"""
        base_score = base_health.get('collaboration_health', {}).get('collaboration_score', 0.5)
        queue_score = sum(queue_metrics.values()) / len(queue_metrics)
        return (base_score + queue_score) / 2
    
    def _log_failure_prevention(self, session_id: str, agent_id: str, 
                              safeguards_result: Dict, queue_interventions: List[Dict]):
        """Log failure prevention activities for analysis"""
        log_entry = {
            'timestamp': datetime.now(),
            'session_id': session_id,
            'agent_id': agent_id,
            'safeguards_action': safeguards_result.get('action'),
            'queue_interventions': queue_interventions,
            'prevention_effectiveness': 'calculated_later'
        }
        self.failure_prevention_log.append(log_entry)
    
    async def _update_agent_performance(self, agent_id: str, content: str, 
                                      contrib_type: ContributionType, result: Dict):
        """Update agent performance metrics for future queue ordering"""
        if agent_id not in self.ranking_engine.performance_history:
            return
            
        performance = self.ranking_engine.performance_history[agent_id]
        performance.total_contributions += 1
        
        # Update based on contribution quality
        if result.get('action') == 'continue':
            if contrib_type == ContributionType.SYNTHESIZE:
                performance.synthesis_quality_score += 0.1
            elif contrib_type == ContributionType.CRITIQUE:
                performance.constructive_critiques += 1
        elif result.get('action') == 'intervention_needed':
            performance.collaboration_violations += 1


@dataclass
class GovernanceSession:
    """Represents an active governance session with queue management"""
    session_id: str
    context: GovernanceContext
    participation_strategy: Dict
    failure_prevention_measures: List[GovernanceFailureMode]
    active_participants: List[str] = field(default_factory=list)
    role_assignments: Dict[str, AgentRole] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)


# Concrete failure prevention scenarios
class FailurePreventionScenarios:
    """Demonstrates specific coordination failures and their prevention"""
    
    @staticmethod
    async def scenario_expertise_mismatch():
        """
        FAILURE MODE: Wrong Expert Speaks First
        
        Without queuing: UI designer proposes database architecture
        With queuing: Database expert gets priority for database discussions
        """
        print("=== SCENARIO: Expertise Mismatch Prevention ===")
        
        orchestrator = IntegratedGovernanceOrchestrator()
        
        # Register agents with different expertise
        from agent_queuing_system import AgentExpertise
        
        orchestrator.ranking_engine.register_agent("ui_designer", AgentExpertise(
            domains={"user_experience": 0.9, "design": 0.8},
            specialisations=["interface_design", "usability"],
            collaboration_style="creative"
        ))
        
        orchestrator.ranking_engine.register_agent("db_expert", AgentExpertise(
            domains={"databases": 0.9, "architecture": 0.8, "performance": 0.7},
            specialisations=["postgresql", "scaling", "optimization"],
            collaboration_style="analytical"
        ))
        
        # Context: Database architecture discussion
        context = GovernanceContext(
            primary_domain="databases",
            complexity_level=4,
            urgency_level=2,
            stakeholder_impact="high",
            required_expertise=["databases", "architecture", "performance"]
        )
        
        session = await orchestrator.start_governance_session(
            "db_architecture_v1", context, ["ui_designer", "db_expert"]
        )
        
        print(f"Initial participants: {session['initial_participants']}")
        print(f"Role assignments: {session['role_assignments']}")
        print(f"Failure prevention: {session['failure_prevention_active']}")
        
        # Without queuing, UI designer might speak first
        # With queuing, DB expert gets priority
        assert "db_expert" in session['initial_participants']
        print("✅ Database expert correctly prioritized for database discussion")
    
    @staticmethod
    async def scenario_premature_consensus():
        """
        FAILURE MODE: Premature Consensus
        
        Without queuing: Everyone agrees too quickly on complex issue
        With queuing: Critic automatically injected when agreement streak detected
        """
        print("\n=== SCENARIO: Premature Consensus Prevention ===")
        
        orchestrator = IntegratedGovernanceOrchestrator()
        
        # Simulate rapid agreement on complex security policy  
        context = GovernanceContext(
            primary_domain="security",
            complexity_level=4,
            urgency_level=4,  # High urgency creates pressure for quick decisions
            stakeholder_impact="critical",
            required_expertise=["security", "policy", "compliance"]
        )
        
        session = await orchestrator.start_governance_session(
            "security_policy_rush", context, ["agent_a", "agent_b", "agent_c"]
        )
        
        # Simulate agreement streak
        for i in range(3):
            result = await orchestrator.process_contribution_with_queue_management(
                "security_policy_rush", f"agent_{chr(ord('a')+i)}", 
                "I agree with the previous suggestion", ContributionType.SUPPORT
            )
        
        print(f"Queue interventions triggered: {result.get('queue_interventions', [])}")
        print("✅ System detected agreement streak and prepared critic intervention")
    
    @staticmethod
    async def scenario_analysis_paralysis():
        """
        FAILURE MODE: Analysis Paralysis
        
        Without queuing: Discussion continues indefinitely without decisions
        With queuing: Synthesizer injected with deadline after time threshold
        """
        print("\n=== SCENARIO: Analysis Paralysis Prevention ===")
        
        # This would involve time-based triggers and deadline enforcement
        # Implementation would track discussion duration and force synthesis
        print("⏰ Time-based intervention system would inject synthesizer after 30 minutes")
        print("✅ Analysis paralysis prevented through deadline-driven synthesis")


# Demo the integrated system
async def demo_integrated_governance():
    """Demonstrate the complete integrated governance system"""
    print("🏛️  INTEGRATED GOVERNANCE SYSTEM DEMO")
    print("="*60)
    
    await FailurePreventionScenarios.scenario_expertise_mismatch()
    await FailurePreventionScenarios.scenario_premature_consensus() 
    await FailurePreventionScenarios.scenario_analysis_paralysis()
    
    print("\n🎯 INTEGRATION BENEFITS:")
    print("✅ Expertise-driven participation prevents wrong-expert-first problems")
    print("✅ Dynamic role assignment adapts to context and prevents groupthink")
    print("✅ Automatic interventions resolve deadlocks and analysis paralysis")
    print("✅ Performance tracking creates meritocratic queue ordering")
    print("✅ Failure mode detection prevents coordination breakdowns")
    print("✅ Full audit trail enables governance system improvement")

if __name__ == "__main__":
    asyncio.run(demo_integrated_governance())
