"""Exercise range transport, expiring URL refresh, integrity, and cached resume."""
import hashlib
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

from scripts.cluster import ranged_download


def test_expired_url_download_and_verified_resume(tmp_path, monkeypatch):
    payload = b'G05 checkpoint transport test\x00\xff' * 4096
    requests_seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            start, end = map(int, self.headers['Range'].removeprefix('bytes=').split('-'))
            requests_seen.append((start, end))
            body = payload[start:end+1]
            self.send_response(206)
            self.send_header('Content-Range', f'bytes {start}-{end}/{len(payload)}')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    metadata_calls = []

    def metadata(url, token):
        metadata_calls.append(url)
        expires = 1 if len(metadata_calls) == 1 else int(time.time())+3600
        return SimpleNamespace(location=f'http://127.0.0.1:{server.server_port}/weights?Expires={expires}')

    monkeypatch.setattr(ranged_download, 'get_hf_file_metadata', metadata)
    entry = SimpleNamespace(path='bundle/model.pt', size=len(payload),
                            lfs=SimpleNamespace(sha256=hashlib.sha256(payload).hexdigest()))
    target = tmp_path / 'checkpoints' / entry.path
    try:
        ranged_download.download_large('owner/model', 'revision', entry, target, workers=2)
        assert target.read_bytes() == payload
        assert len(metadata_calls) == 2  # Expired initial URL was refreshed.
        assert requests_seen == [(0, len(payload)-1)]
        ranged_download.download_large('owner/model', 'revision', entry, target, workers=2)
        assert len(metadata_calls) == 2
        assert len(requests_seen) == 1  # Verified file does not download again.
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
