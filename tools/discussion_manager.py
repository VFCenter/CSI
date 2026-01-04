from typing import Dict, List, Any, Optional
from loguru import logger
import json
import time
class DiscussionManager:
    
    def __init__(self):
        self.discussion_history = []
        self.participants = {}
        self.current_round = 0
        
    def add_participant(self, agent_id: str, agent_name: str, agent_type: str):
        self.participants[agent_id] = {
            'name': agent_name,
            'type': agent_type,
            'contributions': []
        }
        logger.info(f"Added participant: {agent_name} ({agent_type})")
    
    def start_discussion(self, topic: str, initial_data: Dict[str, Any]) -> str:
        self.current_round += 1
        discussion_id = f"discussion_{int(time.time())}_{self.current_round}"
        
        self.discussion_history.append({
            'discussion_id': discussion_id,
            'topic': topic,
            'initial_data': initial_data,
            'rounds': [],
            'participants': list(self.participants.keys()),
            'status': 'active'
        })
        
        logger.info(f"Started discussion: {topic} (ID: {discussion_id})")
        return discussion_id
    
    def add_contribution(self, discussion_id: str, agent_id: str, contribution: str, round_num: int = 1):
        try:
            
            discussion = None
            for disc in self.discussion_history:
                if disc['discussion_id'] == discussion_id:
                    discussion = disc
                    break
            
            if not discussion:
                logger.error(f"Discussion not found: {discussion_id}")
                return False
            
            
            while len(discussion['rounds']) < round_num:
                discussion['rounds'].append([])
            
            
            contribution_data = {
                'agent_id': agent_id,
                'agent_name': self.participants.get(agent_id, {}).get('name', 'Unknown'),
                'contribution': contribution,
                'timestamp': time.time()
            }
            
            discussion['rounds'][round_num - 1].append(contribution_data)
            
            
            if agent_id in self.participants:
                self.participants[agent_id]['contributions'].append(contribution_data)
            
            logger.info(f"Added contribution from {agent_id} to discussion {discussion_id}, round {round_num}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add contribution: {e}")
            return False
    
    def get_discussion_summary(self, discussion_id: str, round_num: Optional[int] = None) -> str:
        try:
            
            discussion = None
            for disc in self.discussion_history:
                if disc['discussion_id'] == discussion_id:
                    discussion = disc
                    break
            
            if not discussion:
                return f"Discussion not found: {discussion_id}"
            
            summary = f"""
=== Discussion Summary ===
Topic: {discussion['topic']}
Discussion ID: {discussion_id}
Participants: {len(discussion['participants'])}