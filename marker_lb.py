#!/usr/bin/env python
"""Least-connections reverse proxy for the marker API.

marker_server does not spread a single process across GPUs, so to use several
GPUs we run one marker_server per GPU (pinned with CUDA_VISIBLE_DEVICES) and put
this proxy in front of them. It exposes one endpoint and forwards each incoming
request to whichever backend currently has the fewest in-flight requests, which
keeps all GPUs busy without a fixed round-robin sending two long jobs to the
same GPU while another sits idle.

Standalone on purpose: depends only on starlette + httpx + uvicorn, all of which
marker already requires. Launched by run_server.sh, or directly:

    python marker_lb.py --host 0.0.0.0 --port 8000 \
        --backends http://127.0.0.1:8001 http://127.0.0.1:8002
"""

import argparse
from contextlib import asynccontextmanager

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

# Hop-by-hop headers (RFC 7230 §6.1) plus framing headers we must not forward;
# httpx/starlette recompute framing for the new connection.
_DROP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
}


def build_app(backends, timeout):
    # in-flight request count per backend, used to pick the least-busy one
    inflight = {b: 0 for b in backends}

    @asynccontextmanager
    async def lifespan(app):
        # One shared client. read=None: a single PDF conversion can run for
        # minutes, so we must not impose a read timeout on the backend response.
        app.state.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, read=None), follow_redirects=False
        )
        yield
        await app.state.client.aclose()

    def pick_backend():
        # min() is stable, so ties keep the first-listed backend — fine here.
        return min(backends, key=lambda b: inflight[b])

    async def health(request: Request):
        return JSONResponse({"backends": dict(inflight)})

    async def proxy(request: Request):
        backend = pick_backend()
        inflight[backend] += 1
        decremented = False

        def release():
            nonlocal decremented
            if not decremented:
                inflight[backend] -= 1
                decremented = True

        try:
            url = backend + request.url.path
            if request.url.query:
                url += "?" + request.url.query
            fwd_headers = {
                k: v for k, v in request.headers.items()
                if k.lower() not in _DROP_HEADERS
            }
            # PDF uploads are bounded (tens of MB); buffering the request body is
            # simpler and safe. The *response* is streamed back below.
            body = await request.body()

            client: httpx.AsyncClient = request.app.state.client
            req = client.build_request(
                request.method, url, headers=fwd_headers, content=body
            )
            resp = await client.send(req, stream=True)
        except Exception as e:
            release()
            return JSONResponse(
                {"success": False, "error": f"proxy: backend {backend} unreachable: {e}"},
                status_code=502,
            )

        async def stream_body():
            try:
                async for chunk in resp.aiter_raw():
                    yield chunk
            finally:
                await resp.aclose()
                release()

        resp_headers = {
            k: v for k, v in resp.headers.items() if k.lower() not in _DROP_HEADERS
        }
        return StreamingResponse(
            stream_body(), status_code=resp.status_code, headers=resp_headers
        )

    routes = [
        Route("/_lb/health", health, methods=["GET"]),
        Route(
            "/{path:path}",
            proxy,
            methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
        ),
    ]
    return Starlette(routes=routes, lifespan=lifespan)


def main():
    parser = argparse.ArgumentParser(description="marker load-balancing proxy")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--backends", nargs="+", required=True,
        help="Backend base URLs, e.g. http://127.0.0.1:8001 http://127.0.0.1:8002",
    )
    parser.add_argument(
        "--timeout", type=float, default=30.0,
        help="Connect/write timeout (s) to backends; response read has no timeout.",
    )
    args = parser.parse_args()

    app = build_app(args.backends, args.timeout)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
