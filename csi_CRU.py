from agents.case_review_unit.review_team.reviewer1 import Reviewer1
from agents.case_review_unit.review_team.reviewer2 import Reviewer2
from agents.case_review_unit.review_team.reviewer3 import Reviewer3
from tools.discussion_manager import DiscussionManager
from util import post_process_csi_result
import concurrent.futures
def execute_deliberation_flow(case_file: str, video_id: str, dataset: str = "FakeTT", max_rounds: int = 3):
    try:
        Agent1 = Reviewer1(temperature=1.0)
        Agent2 = Reviewer2(temperature=1.0)
        Agent3 = Reviewer3(temperature=1.0)
        discussion_manager = DiscussionManager()
        all_discussions = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future1 = executor.submit(Agent1.analyze, case_file, video_id, [])
            future2 = executor.submit(Agent2.analyze, case_file, video_id, [])
            future3 = executor.submit(Agent3.analyze, case_file, video_id, [])
            agent1_output = future1.result()
            agent2_output = future2.result()
            agent3_output = future3.result()
        
        round1_discussions = [
            {'agent': '1', 'role': ' ', 'round': 1, 'output': agent1_output},
            {'agent': '2', 'role': ' ', 'round': 1, 'output': agent2_output},
            {'agent': '3', 'role': ' ', 'round': 1, 'output': agent3_output}
        ]
        all_discussions.extend(round1_discussions)
        
        for discussion in round1_discussions:
            discussion_manager.add_discussion(video_id, discussion)
        
        for current_round in range(2, max_rounds):
            previous_round = current_round - 1
            previous_round_discussions = [d for d in all_discussions if d.get('round') == previous_round]
            meeting_log = generate_meeting_log_for_round(previous_round_discussions, previous_round)
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future1 = executor.submit(Agent1.analyze_with_meeting_log, case_file, video_id, meeting_log)
                future2 = executor.submit(Agent2.analyze_with_meeting_log, case_file, video_id, meeting_log)
                future3 = executor.submit(Agent3.analyze_with_meeting_log, case_file, video_id, meeting_log)
                out1 = future1.result()
                out2 = future2.result()
                out3 = future3.result()
            
            current_round_discussions = [
                {'agent': '1', 'role': ' ', 'round': current_round, 'output': out1},
                {'agent': '2', 'role': ' ', 'round': current_round, 'output': out2},
                {'agent': '3', 'role': ' ', 'round': current_round, 'output': out3}
            ]
            
            all_discussions.extend(current_round_discussions)
            
            for discussion in current_round_discussions:
                discussion_manager.add_discussion(video_id, discussion)
        
        agent1_final_output = Agent1.generate_final_summary(case_file, video_id, all_discussions)
        final_discussion = {
            'agent': '1',
            'role': 'Team Leader',
            'round': max_rounds,
            'output': agent1_final_output
        }
        all_discussions.append(final_discussion)
        discussion_manager.add_discussion(video_id, final_discussion)
        
        final_meeting_log = generate_meeting_log(all_discussions, max_rounds)
        
        processed_result = post_process_csi_result(agent1_final_output, "fake_detection")
        results = {
            "final_decision": agent1_final_output,
            "analysis": processed_result["analysis"],
            "discussion_history": all_discussions,
            "total_rounds": max_rounds,
            "agents_participated": ['1', '2', '3'],
            "meeting_log": final_meeting_log
        }
        return results
    except Exception as e:
        return {
            "final_decision": f"Error: {str(e)}",
            "analysis": "Analysis failed",
            "discussion_history": [],
            "total_rounds": 0,
            "agents_participated": [],
            "meeting_log": ""
        }
def generate_meeting_log_for_round(discussions: list, round_num: int) -> str:
    meeting_log = f"\n=== Meeting Log (Round {round_num}) ===\n\n"
    for discussion in discussions:
        agent = discussion.get('agent', 'Unknown')
        role = discussion.get('role', 'Unknown Role')
        analysis = discussion.get('output', '')
        meeting_log += f"Reviewer {agent} ({role}):\n"
        meeting_log += f"{analysis}\n"
        meeting_log += "-" * 60 + "\n\n"
    return meeting_log

def generate_meeting_log(discussions: list, current_round: int) -> str:
    meeting_log = f"\n=== Meeting Log (Rounds 1-{current_round}) ===\n\n"
    for discussion in discussions:
        round_num = discussion.get('round', 'Unknown')
        agent = discussion.get('agent', 'Unknown')
        role = discussion.get('role', 'Unknown Role')
        analysis = discussion.get('output', '')
        meeting_log += f"Round {round_num} - Agent {agent} ({role}):\n"
        meeting_log += f"{analysis}\n"
        meeting_log += "-" * 60 + "\n\n"
    return meeting_log