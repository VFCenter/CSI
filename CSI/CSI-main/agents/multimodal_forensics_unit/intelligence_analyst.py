import os
import sys
from typing import Dict, Any
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from util import ask_llm_with_tools
from tools.extract_text import extract_text
from tools.text_retrieval_tool import retrieve_google_search
from tools.casefile_manager import update_multimodal_forensics_results
from tools.tool_adapter import ToolAdapter, ToolDefinition
class IntelligenceAnalyst:
    def __init__(self, temperature=0.0, llm_provider: str = None):
        self.temperature = temperature
        self.tool_adapter = ToolAdapter(llm_provider=llm_provider)
        self._setup_tools()
    
    def _setup_tools(self):
        tool_def = ToolDefinition(
            name="retrieve_google_search",
            description="Search Google for related texts and official reports about the video content.",
            parameters={
                "query": {"type": "string", "description": "Search query based on analyzed text"},
                "video_id": {"type": "string", "description": "Video ID for context"},
                "dataset": {"type": "string", "description": "Dataset name (FakeTT or FakeSV)"}
            },
            required=["query", "video_id", "dataset"]
        )
        
        def executor(query: str, video_id: str, dataset: str) -> str:
            return retrieve_google_search(query, video_id, dataset)
        
        self.tool_adapter.register_tool(tool_def, executor)
        self.tool_def = tool_def
    
    def analyze(self, video_id: str, input_data: Dict[str, Any]) -> str:
        try:
            dataset = input_data.get('dataset', 'FakeTT')
            extracted_text = extract_text(video_id, dataset)
            tool_schema = self.tool_adapter.get_tool_schema_for_provider(self.tool_def)
            
            prompt = " "
            
            response = ask_llm_with_tools(
                prompt_text=prompt,
                tools=[tool_schema] if tool_schema else [],
                tool_executor=self.tool_adapter.tool_executors,
                temperature=self.temperature,
                llm_provider=self.tool_adapter.llm_provider
            )
            
            update_multimodal_forensics_results(video_id, text_result=response)
            return response
        except Exception as e:
            error_result = f"Error: {str(e)}"
            update_multimodal_forensics_results(video_id, text_result=error_result)
            return error_result