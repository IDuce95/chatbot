import re
from typing import Any, Dict, List

from ..base_tool import BaseTool


class CodeGeneratorTool(BaseTool):

    def __init__(self, chatbot, config: Dict[str, Any]):
        self.chatbot = chatbot
        self.config = config

    def execute(self, requirements: str, research_context: str = "") -> Dict[str, Any]:
        code_prompt = self.config["agents"]["code"]["prompt"]

        full_context = f"Requirements: {requirements}\n\nContext: {research_context}"

        messages = [
            {"role": "system", "content": code_prompt},
            {"role": "user", "content": full_context}
        ]

        try:
            response = self.chatbot.client.chat.completions.create(
                model=self.chatbot.config["model"]["name"],
                messages=messages,
                max_tokens=2000,
                temperature=0.3
            )

            generated_code = response.choices[0].message.content.strip()
            code_blocks = self._extract_code_blocks(generated_code)

            return {
                'full_response': generated_code,
                'code_blocks': code_blocks,
                'language': 'python',
                'explanation': self._extract_explanation(generated_code)
            }

        except Exception as e:
            print(f"Code generation error: {e}")
            return {
                'full_response': f"Error generating code: {e}",
                'code_blocks': [],
                'language': 'python',
                'explanation': "Code generation failed"
            }

    def _extract_code_blocks(self, text: str) -> List[str]:
        pattern = r'```(?:python)?\n?(.*?)\n?```'
        matches = re.findall(pattern, text, re.DOTALL)
        return [match.strip() for match in matches]

    def _extract_explanation(self, text: str) -> str:
        without_code = re.sub(r'```(?:python)?\n?.*?\n?```', '', text, flags=re.DOTALL)
        return without_code.strip()


class CodeExecutorTool(BaseTool):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sandbox_enabled = config.get("agents", {}).get("code", {}).get("sandbox", False)

    def execute(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Execute code in sandbox environment."""
        if not self.sandbox_enabled:
            return {
                'executed': False,
                'output': "Code execution disabled (sandbox not configured)",
                'error': None,
                'exit_code': -1
            }

        print("🐳 Docker sandbox execution not implemented yet")

        return {
            'executed': False,
            'output': f"Sandbox execution for {language} code would run here:\n{code}",
            'error': None,
            'exit_code': 0,
            'sandbox': 'docker'
        }


class LinterTool(BaseTool):
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def execute(self, code: str, language: str = "python") -> Dict[str, Any]:
        issues = []

        if language == "python":
            issues.extend(self._check_python_basics(code))

        return {
            'language': language,
            'issues': issues,
            'score': self._calculate_quality_score(issues),
            'suggestions': self._generate_suggestions(issues)
        }

    def _check_python_basics(self, code: str) -> List[Dict[str, str]]:
        issues = []
        lines = code.split('\n')

        for i, line in enumerate(lines, 1):
            if len(line) > 100:
                issues.append({
                    'line': i,
                    'type': 'style',
                    'message': 'Line too long (>100 characters)',
                    'severity': 'warning'
                })

            if line.strip().endswith('\\'):
                issues.append({
                    'line': i,
                    'type': 'style',
                    'message': 'Avoid line continuation backslash',
                    'severity': 'info'
                })

        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            issues.append({
                'line': e.lineno or 0,
                'type': 'error',
                'message': f'Syntax error: {e.msg}',
                'severity': 'error'
            })

        return issues

    def _calculate_quality_score(self, issues: List[Dict[str, str]]) -> float:
        if not issues:
            return 10.0

        penalty = 0
        for issue in issues:
            severity = issue.get('severity', 'info')
            if severity == 'error':
                penalty += 3
            elif severity == 'warning':
                penalty += 1
            else:
                penalty += 0.5

        score = max(0, 10 - penalty)
        return round(score, 1)

    def _generate_suggestions(self, issues: List[Dict[str, str]]) -> List[str]:
        suggestions = []

        error_count = sum(1 for issue in issues if issue.get('severity') == 'error')
        warning_count = sum(1 for issue in issues if issue.get('severity') == 'warning')

        if error_count > 0:
            suggestions.append("Fix syntax errors before running the code")

        if warning_count > 0:
            suggestions.append("Address style warnings to improve code quality")

        if len(issues) > 5:
            suggestions.append("Consider breaking down complex code into smaller functions")

        return suggestions
