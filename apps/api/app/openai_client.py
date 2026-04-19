from openai import OpenAI

from .config import settings


class TutorLLM:
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None

    def generate_reply(self, student_text: str, language: str = "en") -> tuple[str, int]:
        if not self.client:
            mock = f"[Mock外语助教] 你说的是：{student_text}\n建议表达：...\n语法要点：..."
            return mock, 0

        response = self.client.responses.create(
            model=settings.openai_model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a friendly extracurricular language tutor for students. "
                        "Correct grammar clearly, explain briefly, and encourage learning."
                    ),
                },
                {"role": "user", "content": f"Language={language}; Text={student_text}"},
            ],
        )
        text = response.output_text
        usage = getattr(response, "usage", None)
        total_tokens = usage.total_tokens if usage else 0
        return text, total_tokens


llm_client = TutorLLM()
