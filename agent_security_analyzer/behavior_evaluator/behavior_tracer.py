from langchain_core.callbacks import BaseCallbackHandler
import json

class trace_callback(BaseCallbackHandler):
    def __init__(self):
        self.trace = []
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        self.trace.append({"type": "llm_start", "prompts": prompts})
    
    def on_llm_end(self, output, **kwargs):
        try:
            output = output.generations[0][0].text
        except:
            output = str(output)
        self.trace.append({"type": "llm_end", "output": output})
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name", "unknown_tool") if serialized else "unknown_tool"
        try:
            parsed_input = json.loads(input_str)
        except:
            parsed_input = {"raw": input_str}
        self.trace.append({"type": "tool_start", "tool": tool_name, "input": parsed_input})

    def on_tool_end(self, output, **kwargs):
        self.trace.append({"type": "tool_end", "output": output})
    
