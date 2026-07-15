import os
import random

from locust import HttpUser, task, between

API_KEY = os.getenv("LOCUST_API_KEY", "")

QUESTIONS = [
    "Who is Abu Hurayra Niloy?",
    "What degree is Abu Hurayra Niloy pursuing?",
    "Which university does Abu Hurayra Niloy attend?",
    "What is Abu Hurayra Niloy's GPA?",
    "What is the key idea behind the Reference Sliding Window Attention mechanism?",
    "How does Unlimited OCR achieve long-horizon parsing under limited memory?",
    "Why is constant KV cache important for efficient inference?",
]


class RagUser(HttpUser):
    wait_time = between(0.5, 2.0)

    @task
    def ask_rag(self):
        question = random.choice(QUESTIONS)
        self.client.post(
            "/rag",
            json={"question": question},
            headers={"X-API-Key": API_KEY},
            name="/rag",
        )
