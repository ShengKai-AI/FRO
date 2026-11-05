"""Seekable HTTP ranges for selecting members without fetching whole archives."""
import io
import time
import requests


class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url, size=None):
        self.url, self.size, self.pos = url, size, 0
        self.session = requests.Session()
        self.transferred = 0
        self.resolved_url = url
        self.cache_start = 0
        self.cache = b''
        if size is None:
            self._range(0, 511)

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        if self.pos < 0: raise ValueError('Negative seek')
        return self.pos

    def _range(self, start, end):
        for attempt in range(7):
            try:
                with self.session.get(self.resolved_url, headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'}, stream=True, timeout=(15, 90)) as r:
                    if r.status_code in (401,403):
                        self.resolved_url = self.url
                    r.raise_for_status()
                    header = r.headers.get('Content-Range', '')
                    if r.status_code != 206 or not header.startswith(f'bytes {start}-'):
                        raise ValueError(f'Server did not honor byte range: HTTP {r.status_code}, {header}')
                    self.resolved_url = r.url
                    reported_size = int(header.split('/')[-1])
                    if self.size is not None and self.size != reported_size:
                        raise ValueError('Remote archive size changed from pinned metadata')
                    self.size = reported_size
                    data = r.content
                    if len(data) != end-start+1: raise IOError('Short range response')
                    self.transferred += len(data)
                    return data
            except (requests.RequestException, IOError):
                if attempt == 6: raise
                time.sleep(min(2**attempt, 20))

    def read(self, size=-1):
        if size < 0: size = self.size-self.pos
        size = min(size, self.size-self.pos)
        if size <= 0: return b''
        if size <= 16384:
            if not (self.cache_start <= self.pos and self.pos+size <= self.cache_start+len(self.cache)):
                self.cache_start = self.pos
                self.cache = self._range(self.pos, min(self.pos+65535,self.size-1))
            start=self.pos-self.cache_start;data=self.cache[start:start+size]
        else:
            data=b''.join(self._range(start,min(start+8*1024*1024-1,self.pos+size-1)) for start in range(self.pos,self.pos+size,8*1024*1024))
        self.pos += len(data)
        return data

    def readinto(self, buffer):
        data = self.read(len(buffer)); buffer[:len(data)] = data; return len(data)

    def close(self):
        self.session.close()
        super().close()
