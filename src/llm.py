"""
Client LLM minimal — 100% gratuit (Groq ou Gemini).

Plomberie fournie : ce n'est pas l'objet de l'apprentissage.
Note : la logique de retry/backoff est ecrite a la main avec tenacity,
pas via un framework — conformement a la regle "no LangChain".
"""
from __future__ import annotations
from tenacity import retry, stop_after_attempt, wait_exponential
from .config import settings


class LLMClient:
    def __init__(self, provider: str | None = None, model: str | None = None):
        self.provider = provider or settings.llm_provider
        self.model = model or settings.llm_model
        self._client = None
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def _get(self):
        if self._client is not None:
            return self._client
        if self.provider == "groq":
            from groq import Groq
            self._client = Groq(api_key=settings.groq_api_key)
        elif self.provider == "gemini":
            from google import genai
            self._client = genai.Client(api_key=settings.google_api_key)
        elif self.provider == "ollama":
            import httpx
            self._client = httpx.Client(base_url="http://localhost:11434", timeout=120)
        else:
            raise ValueError(f"Fournisseur inconnu : {self.provider}")
        return self._client

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=30))
    def complete(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        """ATTENTION — piege verifie le 2026-08-29 :
        `openai/gpt-oss-120b` est un modele A RAISONNEMENT. Il consomme des
        tokens de reflexion AVANT d'ecrire sa reponse. Avec max_tokens trop bas
        (teste : 10), il renvoie une chaine VIDE sans lever d'erreur.
        Ne jamais descendre sous ~256, meme pour une reponse d'un seul mot.
        Alternative sans raisonnement, 3x plus rapide : qwen/qwen3.8-27b."""
        c = self._get()
        if self.provider == "groq":
            r = c.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if r.usage:
                self.total_input_tokens += r.usage.prompt_tokens
                self.total_output_tokens += r.usage.completion_tokens
            return r.choices[0].message.content
        if self.provider == "gemini":
            r = c.models.generate_content(model=self.model, contents=prompt)
            return r.text
        r = c.post("/api/generate", json={"model": self.model, "prompt": prompt, "stream": False})
        return r.json()["response"]
