import asyncio
import websockets
import json

async def test():
    async with websockets.connect("ws://127.0.0.1:8080/ws/agent") as ws:
        print("Connected")
        async for message in ws:
            if isinstance(message, str):
                print(message)
                if json.loads(message).get("type") in ["SettingsApplied", "Error"]:
                    break

asyncio.run(test())
