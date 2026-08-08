"""Entry point — wires transport, protocol, and application layers."""
import signal
import sys
import time

from config import HOST, PORT
from transport.udp_transport import UDPTransport
from protocol.serializer import serialize, deserialize
from application.handler import handle
from utils.logger import log


def main() -> None:
    transport = UDPTransport(HOST, PORT)

    def on_packet(data: bytes, addr: tuple[str, int]) -> None:
        log.info("Received from %s:%d — raw: %r", *addr, data)
        pkt = deserialize(data)
        if pkt is None:
            log.warning("Discarding malformed packet from %s", addr)
            return
        log.info("Parsed: %s", pkt.type.name)
        reply = handle(pkt)
        if reply is not None:
            transport.send_bytes(serialize(reply))

    def _shutdown(_signum, _frame):
        transport.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    transport.start()
    transport.receive_bytes(on_packet)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        transport.stop()


if __name__ == "__main__":
    main()
