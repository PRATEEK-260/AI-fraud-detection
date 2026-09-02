"""Minimal async pub/sub connecting agents to consumers.

Agents publish Case objects; the dashboard and adjudicator subscribe.
An asyncio.Queue per subscriber is all this buildathon needs — no Kafka.
"""

import asyncio

from spine.schema import Case


class EventBus:
    def __init__(self) -> None:
        self._queues: list[asyncio.Queue] = []

    def subscribe(self, maxsize: int = 1000) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._queues.append(q)
        return q

    async def publish(self, case: Case) -> None:
        for q in self._queues:
            if q.full():
                # slow consumer: drop oldest rather than block the pipeline
                q.get_nowait()
            q.put_nowait(case)

    async def publish_many(self, cases: list[Case]) -> None:
        for case in cases:
            await self.publish(case)
