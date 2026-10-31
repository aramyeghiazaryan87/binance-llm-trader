"""Контракт с настоящим SDK Anthropic: запрос уходит в правильном формате, ответ корректно разбирается.

Сеть не используется: HTTP подменён MockTransport.
"""
import json

import anthropic
import httpx2
from anthropic import DefaultHttpxClient

from trader.budget import BudgetGuard
from trader.llm import FALLBACK_BETA, LLMAdvisor, MarketAssessment
from trader.storage import Storage

from .test_llm import make_ctx


def test_real_sdk_request_shape_and_parsing(now):
    captured = {}

    def handler(request):
        captured["headers"] = dict(request.headers)
        captured["body"] = body = json.loads(request.content)
        prompt = body["messages"][0]["content"]
        sig_ids = [line.split("id=")[1].split()[0] for line in prompt.splitlines() if line.startswith("- id=")]
        answer = MarketAssessment(
            market_regime="neutral",
            news_sentiment=0.1,
            news_summary="quiet",
            watchlist=[{"symbol": "SOL/USDT", "reason": "breakout"}],
            signal_verdicts=[
                {"signal_id": s, "approve": True, "confidence": 0.75, "size_factor": 1.0, "reason": "ok"} for s in sig_ids
            ],
            position_advice=[],
            report="Рынок спокойный.",
        )
        return httpx2.Response(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-5-5",
                "content": [{"type": "text", "text": answer.model_dump_json()}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1834, "output_tokens": 612},
            },
        )

    client = anthropic.Anthropic(
        api_key="sk-test", max_retries=0, http_client=DefaultHttpxClient(transport=httpx2.MockTransport(handler))
    )
    storage = Storage(":memory:")
    advisor = LLMAdvisor(storage=storage, budget=BudgetGuard(storage, 10.0), model="claude-sonnet-5-5",
                         effort="low", max_tokens=4000, max_watchlist=5, client=client)
    ctx, sig = make_ctx(now)
    result = advisor.decide(ctx)

    body = captured["body"]
    assert captured["headers"]["anthropic-beta"] == FALLBACK_BETA
    assert body["model"] == "claude-sonnet-5-5"
    assert body["fallbacks"] == "default"
    assert body["output_config"]["effort"] == "low"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert "temperature" not in body and "thinking" not in body
    assert result.ok
    assert result.assessment.signal_verdicts[0].signal_id == sig.id
    assert result.cost_usd > 0
