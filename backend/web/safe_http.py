import ipaddress
import socket
from collections.abc import Callable
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

import httpx

Resolver = Callable[[str, int], list[str]]

_ALLOWED_SCHEMES = {"http": 80, "https": 443}
_ALLOWED_CONTENT_TYPES = ("text/html", "application/xhtml+xml", "text/plain")
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}
_USER_AGENT = "HomelabAI/0.1 (+local research assistant)"

_SKIPPED_TAGS = {"script", "style", "noscript", "template", "svg", "nav", "footer", "aside", "form"}
_BLOCK_TAGS = {
    "p",
    "div",
    "br",
    "li",
    "tr",
    "section",
    "article",
    "pre",
    "blockquote",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
}


class UnsafeURLError(ValueError):
    """The URL is not a public http(s) address and must not be requested."""


class FetchError(RuntimeError):
    """The page could not be retrieved or was not usable."""


@dataclass(frozen=True)
class FetchedPage:
    url: str
    title: str
    text: str


@dataclass(frozen=True)
class ResolvedTarget:
    scheme: str
    host: str
    port: int
    path: str
    address: str

    @property
    def host_header(self) -> str:
        default_port = _ALLOWED_SCHEMES[self.scheme]
        host = f"[{self.host}]" if ":" in self.host else self.host
        return host if self.port == default_port else f"{host}:{self.port}"

    @property
    def pinned_url(self) -> str:
        address = f"[{self.address}]" if ":" in self.address else self.address
        return f"{self.scheme}://{address}:{self.port}{self.path}"


def default_resolver(host: str, port: int) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as error:
        raise FetchError(f"could not resolve {host}") from error
    return [str(info[4][0]) for info in infos]


def is_public_address(address: str) -> bool:
    ip = ipaddress.ip_address(address.split("%", 1)[0])
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


def resolve_public_url(url: str, resolver: Resolver = default_resolver) -> ResolvedTarget:
    """
    Validate a URL and resolve it to a public address.

    Rejects non-http(s) schemes, embedded credentials, and any host with even one
    private, loopback, link-local or otherwise non-public address.
    """

    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise UnsafeURLError("only http and https URLs are allowed")
    if parts.username is not None or parts.password is not None:
        raise UnsafeURLError("URLs with embedded credentials are not allowed")

    host = parts.hostname
    if not host:
        raise UnsafeURLError("the URL has no host")
    try:
        port = parts.port or _ALLOWED_SCHEMES[scheme]
    except ValueError as error:
        raise UnsafeURLError("the URL has an invalid port") from error

    try:
        ipaddress.ip_address(host)
        addresses = [host]
    except ValueError:
        addresses = resolver(host, port)
    if not addresses:
        raise FetchError(f"could not resolve {host}")

    for address in addresses:
        if not is_public_address(address):
            raise UnsafeURLError(f"{host} resolves to a non-public address")

    path = parts.path or "/"
    if parts.query:
        path = f"{path}?{parts.query}"
    return ResolvedTarget(scheme, host, port, path, addresses[0])


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self._parts: list[str] = []
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "title":
            self._in_title = True
        elif tag in _SKIPPED_TAGS:
            self._skip_depth += 1
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        elif tag in _SKIPPED_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        elif self._skip_depth == 0:
            self._parts.append(data)

    def text(self) -> str:
        lines = (" ".join(line.split()) for line in "".join(self._parts).splitlines())
        return "\n".join(line for line in lines if line)


def extract_text(html: str) -> tuple[str, str]:
    extractor = _TextExtractor()
    extractor.feed(html)
    extractor.close()
    return " ".join(extractor.title.split()), extractor.text()


class PageFetcher:
    """
    Downloads a public web page and reduces it to plain text.

    Every request goes to an address this class resolved and validated itself, with
    the original hostname sent as the Host header and TLS server name, so a hostname
    cannot change to a private address between the check and the connection. Redirects
    are followed manually and re-validated at each hop.
    """

    def __init__(
        self,
        timeout: float = 8.0,
        max_bytes: int = 1_000_000,
        max_chars: int = 12_000,
        max_redirects: int = 3,
        resolver: Resolver = default_resolver,
        client: httpx.Client | None = None,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if max_bytes <= 0 or max_chars <= 0:
            raise ValueError("max_bytes and max_chars must be positive")
        if max_redirects < 0:
            raise ValueError("max_redirects cannot be negative")

        self._max_bytes = max_bytes
        self._max_chars = max_chars
        self._max_redirects = max_redirects
        self._resolver = resolver
        self._client = client or httpx.Client(
            timeout=timeout,
            follow_redirects=False,
            headers={"User-Agent": _USER_AGENT},
        )

    def fetch(self, url: str) -> FetchedPage:
        current = url
        for _ in range(self._max_redirects + 1):
            target = resolve_public_url(current, self._resolver)
            try:
                with self._client.stream(
                    "GET",
                    target.pinned_url,
                    headers={"Host": target.host_header, "Accept": "text/html,text/plain;q=0.9"},
                    extensions={"sni_hostname": target.host},
                ) as response:
                    if response.status_code in _REDIRECT_STATUSES:
                        location = response.headers.get("location")
                        if not location:
                            raise FetchError("redirect without a location")
                        current = urljoin(current, location)
                        continue
                    return self._read_page(current, response)
            except httpx.HTTPError as error:
                raise FetchError(f"request failed: {error}") from error
        raise FetchError("too many redirects")

    def _read_page(self, url: str, response: httpx.Response) -> FetchedPage:
        if response.status_code >= 400:
            raise FetchError(f"the server answered {response.status_code}")

        content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
        if content_type not in _ALLOWED_CONTENT_TYPES:
            raise FetchError(f"unsupported content type {content_type or 'unknown'}")

        body = bytearray()
        for chunk in response.iter_bytes():
            body.extend(chunk)
            if len(body) >= self._max_bytes:
                del body[self._max_bytes :]
                break

        try:
            text = bytes(body).decode(response.charset_encoding or "utf-8", errors="replace")
        except LookupError:
            text = bytes(body).decode("utf-8", errors="replace")
        if content_type == "text/plain":
            title, content = (
                "",
                "\n".join(line.strip() for line in text.splitlines() if line.strip()),
            )
        else:
            title, content = extract_text(text)
        if not content:
            raise FetchError("the page contained no readable text")
        return FetchedPage(url=url, title=title, text=content[: self._max_chars])
