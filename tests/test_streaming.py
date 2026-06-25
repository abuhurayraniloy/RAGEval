import pytest
from httpx import ASGITransport, AsyncClient
from unittest.mock import patch

from src.main import app


@pytest.mark.asyncio
async def test_streaming():

    async def fake_response():
        for content in ("Explain", " AI"):
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
                                        "content": content
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
                    "prompt": "Explain AI"
                }
            )


    assert response.status_code == 200

    text = response.text

    assert text == "Explain AI"
