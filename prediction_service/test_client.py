import asyncio
import json
import websockets

# IP = "0.0.0.0"
IP = "20.77.139.107"
PORT = 80


async def main() -> None:
    async with websockets.connect(f"ws://{IP}:{PORT}", ping_interval=1000, ping_timeout=1000) as websocket:
        print("Connected to server")

        # Send the initial message to the server
        initial_message = {"type": "connect",
                           "dataset": "sample-1"}  # sample-1, sample-2
        await websocket.send(json.dumps(initial_message))

        message_1 = {"type": "trade",
                     "alpha": "pyraformer",  # informer,transformer, pyraformer
                     "trading_decision_making": "heuristics"}  # heuristics, drl
        await websocket.send(json.dumps(message_1))

        # await asyncio.sleep(35)
        # message_2 = {"type": "change_dataset", "dataset": "sample-2"}
        # await websocket.send(json.dumps(message_2))

        # message_3 = {"type": "stop"}
        # await websocket.send(json.dumps(message_3))

        count = 0
        while True:
            # Receive data from the server
            data = await websocket.recv()  # blocking call
            print('Received', data.rstrip())

            # Send a message to the server while receiving data
            # message_to_server = {"type": "message", "content": "Hello, server!"}
            # await websocket.send(json.dumps(message_to_server))

            await asyncio.sleep(0.1)
            count = count + 1

            # if count == 20:
            #    print("stopping !")
            #    message_2 = {"type": "stop_trading"}
            #    await websocket.send(json.dumps(message_2))


if __name__ == '__main__':
    asyncio.run(main())


def save_json(data: bytes) -> None:
    """Saves json packets to file and name it with id"""
    filename = ''.join(
        random.choice(string.digits)
        for _ in range(3)
    ) + '.json'
    with open(filename, 'wb') as f:
        f.write(data)
    print('saved to', filename)


def generate_json_message() -> Dict[str, Any]:
    # """Generate random json packet with hashed data bits"""
    return {
        "id": randint(1, 100),
        "timestamp": time(),
        "data": hash(str(randint(1, 100)))
    }


async def send_json_message(
    websocket: websockets.WebSocketClientProtocol,
    json_message: Dict[str, Any],
) -> None:
    """Send json packet to server"""
    message = (json.dumps(json_message) + '\n').encode()
    await websocket.send(message)
    print(f'{len(message)} bytes sent')


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
