"""Resumable bounded HTTP range downloads, verified against Hub LFS SHA-256."""
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from huggingface_hub import get_hf_file_metadata, hf_hub_url


def download_large(repo, revision, entry, destination, workers=24):
    destination = Path(destination)
    expected_hash = entry.lfs.sha256
    shard_dir = destination.parents[len(Path(entry.path).parts)-1] / '.ranges' / expected_hash
    shard_dir.mkdir(parents=True, exist_ok=True)
    receipt = shard_dir / 'verified'
    if destination.is_file() and destination.stat().st_size == entry.size and receipt.exists():
        print('Already verified:', entry.path, flush=True)
        return
    metadata = get_hf_file_metadata(hf_hub_url(repo, entry.path, revision=revision), token=True)
    chunk_size = 32 * 1024**2
    ranges = [(i, start, min(start+chunk_size, entry.size)-1)
              for i, start in enumerate(range(0, entry.size, chunk_size))]

    def fetch(part):
        i, start, end = part
        target = shard_dir / f'{i:05d}.part'
        if target.is_file() and target.stat().st_size == end-start+1:
            return target
        temporary = target.with_suffix('.incomplete')
        for attempt in range(4):
            try:
                with requests.get(metadata.location, headers={'Range': f'bytes={start}-{end}'},
                                  stream=True, timeout=(30, 180)) as response:
                    response.raise_for_status()
                    expected_range = f'bytes {start}-{end}/{entry.size}'
                    if response.status_code != 206 or response.headers.get('Content-Range') != expected_range:
                        raise ValueError('Server did not honor requested byte range')
                    with temporary.open('wb') as stream:
                        for data in response.iter_content(1024**2):
                            stream.write(data)
                if temporary.stat().st_size != end-start+1:
                    raise ValueError('Incomplete byte range')
                temporary.replace(target)
                return target
            except (requests.RequestException, ValueError) as exc:
                # Signed URLs contain temporary credentials: never print them.
                print(f'{entry.path} part {i}: {type(exc).__name__}, attempt {attempt+1}', flush=True)
                if attempt == 3:
                    raise RuntimeError(f'Range download failed: {entry.path}, part {i}') from None
                time.sleep(2**attempt)

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, part) for part in ranges]
        for count, future in enumerate(as_completed(futures), 1):
            future.result()
            if count % 16 == 0 or count == len(ranges):
                print(f'{entry.path}: {count}/{len(ranges)} ranges, elapsed {time.monotonic()-started:.0f}s', flush=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    assembled = destination.with_suffix('.assembling')
    digest = hashlib.sha256()
    with assembled.open('wb') as stream:
        for i, _, _ in ranges:
            with (shard_dir / f'{i:05d}.part').open('rb') as part:
                while data := part.read(4*1024**2):
                    digest.update(data)
                    stream.write(data)
    if digest.hexdigest() != expected_hash:
        raise RuntimeError(f'SHA-256 mismatch: {entry.path}; retained shards for diagnosis')
    assembled.replace(destination)
    receipt.write_text(expected_hash + '\n')
    print('SHA-256 verified:', entry.path, flush=True)
