from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.main import app


@pytest.mark.asyncio
async def test_completion():

    async def fake_response():
        yield type(
            "Chunk",
            (),
            {
                "choices": [
                    type(
                        "Choice",
                        (),
                        {
                            "delta": type(
                                "Delta",
                                (),
                                {
                                    "content": "Hello from fake LLM"
                                }
                            )
                        }
                    )
                ]
            }
        )


    with patch(
        "src.main.acompletion",
        return_value=fake_response()
    ):

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test"
        ) as client:

            response = await client.post(
                "/complete",
                json={
                    "prompt": "hello"
                }
            )


    assert response.status_code == 200

    assert response.text == "Hello from fake LLM"
