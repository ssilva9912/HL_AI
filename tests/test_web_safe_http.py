import httpx
import pytest

from backend.web.safe_http import (
    FetchError,
    PageFetcher,
    UnsafeURLError,
    extract_text,
    is_public_address,
    resolve_public_url,
)

PUBLIC_IP = "93.184.216.34"


def resolver_for(mapping: dict[str, list[str]]):
    def resolve(host: str, port: int) -> list[str]:
        del port
        return mapping[host]

    return resolve


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.5",
        "172.16.4.1",
        "192.168.1.10",
        "169.254.169.254",
        "100.64.0.1",
        "0.0.0.0",
        "::1",
        "fe80::1",
        "fc00::1",
        "::ffff:127.0.0.1",
        "::ffff:10.1.2.3",
        "224.0.0.1",
    ],
)
def test_non_public_addresses_are_rejected(address: str) -> None:
    assert not is_public_address(address)


@pytest.mark.parametrize("address", [PUBLIC_IP, "8.8.8.8", "2606:2800:220:1:248:1893:25c8:1946"])
def test_public_addresses_are_accepted(address: str) -> None:
    assert is_public_address(address)


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com/file",
        "file:///etc/passwd",
        "javascript:alert(1)",
        "http://user:pass@example.com/",
        "http:///nohost",
        "http://example.com:notaport/",
        "http://127.0.0.1/admin",
        "http://[::1]/",
        "http://169.254.169.254/latest/meta-data/",
        "http://192.168.0.1:8080/",
    ],
)
def test_unsafe_urls_are_rejected_before_any_request(url: str) -> None:
    with pytest.raises(UnsafeURLError):
        resolve_public_url(url, resolver_for({}))


def test_hostname_resolving_to_a_private_address_is_rejected() -> None:
    with pytest.raises(UnsafeURLError, match="non-public"):
        resolve_public_url("http://evil.test/", resolver_for({"evil.test": ["10.0.0.7"]}))


def test_one_private_record_among_public_ones_is_enough_to_reject() -> None:
    resolver = resolver_for({"mixed.test": [PUBLIC_IP, "192.168.1.1"]})

    with pytest.raises(UnsafeURLError):
        resolve_public_url("https://mixed.test/", resolver)


def test_public_url_is_pinned_to_its_resolved_address() -> None:
    target = resolve_public_url(
        "https://Example.com:8443/a/b?q=1",
        resolver_for({"example.com": [PUBLIC_IP]}),
    )

    assert target.pinned_url == f"https://{PUBLIC_IP}:8443/a/b?q=1"
    assert target.host_header == "example.com:8443"


def test_default_port_is_omitted_from_the_host_header() -> None:
    target = resolve_public_url("http://example.com", resolver_for({"example.com": [PUBLIC_IP]}))

    assert target.host_header == "example.com"
    assert target.pinned_url == f"http://{PUBLIC_IP}:80/"


def test_extract_text_drops_scripts_navigation_and_keeps_headings() -> None:
    html = """
    <html><head><title> My  Page </title><style>p {color: red}</style></head>
    <body><nav>Menu Home</nav><h1>Heading</h1>
    <p>First <b>bold</b> paragraph &amp; more.</p>
    <script>alert('x')</script><footer>Copyright</footer>
    <div>Second block</div></body></html>
    """

    title, text = extract_text(html)

    assert title == "My Page"
    assert text == "Heading\nFirst bold paragraph & more.\nSecond block"


def make_fetcher(handler, **options) -> PageFetcher:
    return PageFetcher(
        resolver=resolver_for(
            {
                "example.com": [PUBLIC_IP],
                "other.example": ["8.8.8.8"],
                "internal.test": ["10.0.0.9"],
            }
        ),
        client=httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False),
        **options,
    )


def html_response(body: str, content_type: str = "text/html; charset=utf-8") -> httpx.Response:
    return httpx.Response(200, content=body.encode(), headers={"content-type": content_type})


def test_fetch_connects_to_the_validated_address_with_the_original_host() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return html_response("<html><title>T</title><body><p>Hello world</p></body></html>")

    page = make_fetcher(handler).fetch("https://example.com/docs?x=1")

    assert page.title == "T"
    assert page.text == "Hello world"
    assert page.url == "https://example.com/docs?x=1"
    assert (seen[0].url.scheme, seen[0].url.host, seen[0].url.port) == ("https", PUBLIC_IP, None)
    assert seen[0].url.raw_path == b"/docs?x=1"
    assert seen[0].headers["host"] == "example.com"
    assert seen[0].extensions["sni_hostname"] == "example.com"


def test_redirect_to_a_private_host_is_blocked() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "http://internal.test/secret"})

    with pytest.raises(UnsafeURLError):
        make_fetcher(handler).fetch("https://example.com/")


def test_redirect_to_a_loopback_literal_is_blocked() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(301, headers={"location": "http://127.0.0.1:8000/health"})

    with pytest.raises(UnsafeURLError):
        make_fetcher(handler).fetch("https://example.com/")


def test_redirects_to_public_hosts_are_followed_with_relative_locations() -> None:
    hosts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hosts.append(request.headers["host"])
        if request.headers["host"] == "example.com":
            return httpx.Response(302, headers={"location": "https://other.example/final"})
        return html_response("<p>Arrived</p>")

    page = make_fetcher(handler).fetch("https://example.com/start")

    assert page.text == "Arrived"
    assert page.url == "https://other.example/final"
    assert hosts == ["example.com", "other.example"]


def test_redirect_loops_are_cut_off() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "https://example.com/again"})

    with pytest.raises(FetchError, match="too many redirects"):
        make_fetcher(handler, max_redirects=2).fetch("https://example.com/")


def test_redirect_without_location_fails() -> None:
    with pytest.raises(FetchError, match="location"):
        make_fetcher(lambda request: httpx.Response(302)).fetch("https://example.com/")


@pytest.mark.parametrize("status", [403, 404, 500])
def test_error_statuses_fail(status: int) -> None:
    with pytest.raises(FetchError, match=str(status)):
        make_fetcher(lambda request: httpx.Response(status)).fetch("https://example.com/")


def test_non_text_content_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"%PDF", headers={"content-type": "application/pdf"})

    with pytest.raises(FetchError, match="unsupported content type"):
        make_fetcher(handler).fetch("https://example.com/file.pdf")


def test_oversized_bodies_are_truncated_not_fully_read() -> None:
    body = "<p>" + "word " * 100_000 + "</p>"

    page = make_fetcher(lambda request: html_response(body), max_bytes=2_000).fetch(
        "https://example.com/"
    )

    assert 0 < len(page.text) <= 2_000


def test_extracted_text_is_capped_at_max_chars() -> None:
    page = make_fetcher(
        lambda request: html_response("<p>" + "a" * 5_000 + "</p>"),
        max_chars=100,
    ).fetch("https://example.com/")

    assert len(page.text) == 100


def test_plain_text_pages_are_supported() -> None:
    page = make_fetcher(
        lambda request: html_response("  line one \n\n line two ", "text/plain")
    ).fetch("https://example.com/notes.txt")

    assert page.text == "line one\nline two"


def test_page_without_readable_text_fails() -> None:
    with pytest.raises(FetchError, match="no readable text"):
        make_fetcher(lambda request: html_response("<script>1</script>")).fetch(
            "https://example.com/"
        )


def test_transport_errors_become_fetch_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("slow", request=request)

    with pytest.raises(FetchError, match="request failed"):
        make_fetcher(handler).fetch("https://example.com/")


def test_unresolvable_host_fails_cleanly() -> None:
    def failing_resolver(host: str, port: int) -> list[str]:
        raise FetchError(f"could not resolve {host}")

    fetcher = PageFetcher(resolver=failing_resolver)

    with pytest.raises(FetchError, match="could not resolve"):
        fetcher.fetch("https://does-not-exist.test/")
