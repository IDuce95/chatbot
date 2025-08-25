import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


class StepTracker:
    def __init__(self):
        self.steps = []
        self.current_step = 0

    def add_step(self, agent_name: str, action: str, details: str = ""):
        self.current_step += 1
        timestamp = __import__('time').time()
        step = {
            "step": self.current_step,
            "agent": agent_name,
            "action": action,
            "details": details,
            "timestamp": timestamp,
            "completed": False
        }
        self.steps.append(step)
        return len(self.steps) - 1

    def complete_step(self, step_index: int, result: str = ""):
        if 0 <= step_index < len(self.steps):
            self.steps[step_index]["completed"] = True
            self.steps[step_index]["result"] = result

    def get_formatted_steps(self) -> list:
        icons = {
            "Router": "🎯",
            "Research": "📚",
            "Code": "💻",
            "Reviewer": "✅",
            "Direct": "🔄"
        }

        formatted = []
        for step in self.steps:
            icon = icons.get(step["agent"], "🔧")
            status = "✅" if step["completed"] else "⏳"

            step_text = f"{status} {icon} {step['agent']}: {step['action']}"
            if step.get("details"):
                step_text += f" ({step['details']})"
            if step.get("result"):
                step_text += f" → {step['result']}"

            formatted.append(step_text)

        return formatted
