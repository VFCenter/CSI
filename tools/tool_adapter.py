import os
from typing import Dict, Any, List, Callable, Optional
from loguru import logger

class ToolDefinition:
    def __init__(self, name: str, description: str, parameters: Dict[str, Any], required: List[str] = None):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.required = required or []

class ToolAdapter:
    def __init__(self, llm_provider: str = None):
        self.llm_provider = llm_provider or os.getenv("LLM_PROVIDER", "gemini")
        self.tool_executors: Dict[str, Callable] = {}
    
    def register_tool(self, tool_def: ToolDefinition, executor: Callable):
        self.tool_executors[tool_def.name] = executor
        return tool_def
    
    def get_tool_schema_for_provider(self, tool_def: ToolDefinition) -> Any:
        if self.llm_provider.lower() == "gemini":
            return self._get_gemini_tool_schema(tool_def)
        elif self.llm_provider.lower() == "openai":
            return self._get_openai_tool_schema(tool_def)
        else:
            logger.warning(f"Unknown LLM provider: {self.llm_provider}, using Gemini format")
            return self._get_gemini_tool_schema(tool_def)
    
    def _get_gemini_tool_schema(self, tool_def: ToolDefinition) -> Any:
        try:
            from google.genai import types
            
            properties = {}
            for param_name, param_info in tool_def.parameters.items():
                param_type = param_info.get('type', 'string')
                gemini_type = types.Type.STRING
                if param_type == 'integer' or param_type == 'number':
                    gemini_type = types.Type.NUMBER
                elif param_type == 'boolean':
                    gemini_type = types.Type.BOOLEAN
                
                properties[param_name] = types.Schema(
                    type=gemini_type,
                    description=param_info.get('description', '')
                )
            
            return types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name=tool_def.name,
                        description=tool_def.description,
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties=properties,
                            required=tool_def.required
                        )
                    )
                ]
            )
        except ImportError:
            logger.error("Google GenAI not available")
            return None
    
    def _get_openai_tool_schema(self, tool_def: ToolDefinition) -> Dict[str, Any]:
        properties = {}
        for param_name, param_info in tool_def.parameters.items():
            properties[param_name] = {
                "type": param_info.get('type', 'string'),
                "description": param_info.get('description', '')
            }
        
        return {
            "type": "function",
            "function": {
                "name": tool_def.name,
                "description": tool_def.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": tool_def.required
                }
            }
        }
    
    def execute_tool(self, tool_name: str, **kwargs) -> str:
        if tool_name in self.tool_executors:
            try:
                logger.info(f"Executing tool: {tool_name} with args: {kwargs}")
                result = self.tool_executors[tool_name](**kwargs)
                return result if isinstance(result, str) else str(result)
            except Exception as e:
                logger.error(f"Error executing tool {tool_name}: {e}")
                return f"Error: {str(e)}"
        else:
            logger.warning(f"Tool {tool_name} not found in tool_executors")
            return f"Tool {tool_name} not available"

