"""
Tests for the ID-JAG (Identity Assertion Authorization Grant) handler for MCP servers.

Covers: the two-legged exchange flow, private-key-JWT vs client_secret client
authentication, caching, error handling, resolve_mcp_auth integration, config/DB
loading, and the has_id_jag_config property.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from litellm.proxy._experimental.mcp_server.auth.id_jag import (
    CLIENT_ASSERTION_TYPE,
    DEFAULT_ID_JAG_SUBJECT_TOKEN_TYPE,
    ID_JAG_REQUESTED_TOKEN_TYPE,
    JWT_BEARER_GRANT_TYPE,
    TOKEN_EXCHANGE_GRANT_TYPE,
    IdJagHandler,
)
from litellm.proxy._experimental.mcp_server.mcp_server_manager import MCPServerManager
from litellm.proxy._experimental.mcp_server.oauth2_token_cache import resolve_mcp_auth
from litellm.proxy._types import LiteLLM_MCPServerTable, MCPTransport
from litellm.types.mcp import MCPAuth
from litellm.types.mcp_server.mcp_server_manager import MCPServer

_RSA_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_PRIVATE_PEM = _RSA_KEY.private_bytes(
    serialization.Encoding.PEM,
    serialization.PrivateFormat.PKCS8,
    serialization.NoEncryption(),
).decode()
_PUBLIC_PEM = (
    _RSA_KEY.public_key()
    .public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    .decode()
)

_LEG1_ENDPOINT = "https://idp.example.com/oauth2/token"
_LEG2_ENDPOINT = "https://mcp-as.example.com/oauth2/token"


def _id_jag_server(**overrides) -> MCPServer:
    defaults = dict(
        server_id="srv-idjag-1",
        name="test-idjag",
        url="https://mcp.example.com/mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2_id_jag,
        client_id="litellm-client-id",
        client_secret="litellm-client-secret",
        token_exchange_endpoint=_LEG1_ENDPOINT,
        id_jag_resource_token_endpoint=_LEG2_ENDPOINT,
        audience="api://mcp-server",
        scopes=["mcp.tools.read", "mcp.tools.execute"],
    )
    defaults.update(overrides)
    return MCPServer(**defaults)


def _resp(token, expires_in=3600):
    resp = MagicMock()
    resp.json.return_value = {
        "access_token": token,
        "token_type": "Bearer",
        "expires_in": expires_in,
    }
    resp.raise_for_status = MagicMock()
    resp.text = ""
    return resp


def _leg(mock_client, index):
    call = mock_client.post.call_args_list[index]
    return call.args[0], call.kwargs["data"]


# ── Two-leg flow ──


@pytest.mark.asyncio
async def test_two_leg_flow():
    """Leg 1 mints the ID-JAG; that assertion is forwarded verbatim to leg 2,
    whose access_token is returned (not the ID-JAG)."""
    handler = IdJagHandler()
    server = _id_jag_server()
    mock_client = AsyncMock()
    mock_client.post.side_effect = [
        _resp("the-id-jag-assertion"),
        _resp("final-mcp-access-token"),
    ]

    with patch(
        "litellm.proxy._experimental.mcp_server.auth.id_jag.get_async_httpx_client",
        return_value=mock_client,
    ):
        result = await handler.exchange_token("user-id-token", server)

    assert result == "final-mcp-access-token"
    assert mock_client.post.call_count == 2

    leg1_url, leg1_data = _leg(mock_client, 0)
    assert leg1_url == _LEG1_ENDPOINT
    assert leg1_data["grant_type"] == TOKEN_EXCHANGE_GRANT_TYPE
    assert leg1_data["requested_token_type"] == ID_JAG_REQUESTED_TOKEN_TYPE
    assert leg1_data["subject_token"] == "user-id-token"
    assert leg1_data["subject_token_type"] == DEFAULT_ID_JAG_SUBJECT_TOKEN_TYPE
    assert leg1_data["audience"] == "api://mcp-server"
    assert leg1_data["scope"] == "mcp.tools.read mcp.tools.execute"

    leg2_url, leg2_data = _leg(mock_client, 1)
    assert leg2_url == _LEG2_ENDPOINT
    assert leg2_data["grant_type"] == JWT_BEARER_GRANT_TYPE
    assert leg2_data["assertion"] == "the-id-jag-assertion"


@pytest.mark.asyncio
async def test_subject_token_type_override_honored():
    """An explicitly configured subject_token_type overrides the id_token default."""
    handler = IdJagHandler()
    server = _id_jag_server(subject_token_type="urn:ietf:params:oauth:token-type:saml2")
    mock_client = AsyncMock()
    mock_client.post.side_effect = [_resp("idjag"), _resp("access")]

    with patch(
        "litellm.proxy._experimental.mcp_server.auth.id_jag.get_async_httpx_client",
        return_value=mock_client,
    ):
        await handler.exchange_token("user-id-token", server)

    _, leg1_data = _leg(mock_client, 0)
    assert leg1_data["subject_token_type"] == "urn:ietf:params:oauth:token-type:saml2"


@pytest.mark.asyncio
async def test_optional_resource_and_scope_omitted_when_absent():
    """resource and scope are omitted from leg 1 when not configured."""
    handler = IdJagHandler()
    server = _id_jag_server(scopes=None, id_jag_resource=None)
    mock_client = AsyncMock()
    mock_client.post.side_effect = [_resp("idjag"), _resp("access")]

    with patch(
        "litellm.proxy._experimental.mcp_server.auth.id_jag.get_async_httpx_client",
        return_value=mock_client,
    ):
        await handler.exchange_token("user-id-token", server)

    _, leg1_data = _leg(mock_client, 0)
    assert "scope" not in leg1_data
    assert "resource" not in leg1_data


@pytest.mark.asyncio
async def test_resource_indicator_forwarded():
    """id_jag_resource is sent as the RFC 8707 resource on leg 1."""
    handler = IdJagHandler()
    server = _id_jag_server(id_jag_resource="https://mcp.example.com/")
    mock_client = AsyncMock()
    mock_client.post.side_effect = [_resp("idjag"), _resp("access")]

    with patch(
        "litellm.proxy._experimental.mcp_server.auth.id_jag.get_async_httpx_client",
        return_value=mock_client,
    ):
        await handler.exchange_token("user-id-token", server)

    _, leg1_data = _leg(mock_client, 0)
    assert leg1_data["resource"] == "https://mcp.example.com/"


# ── Client authentication ──


@pytest.mark.asyncio
async def test_private_key_jwt_client_assertion():
    """With a private key, both legs authenticate via a signed client_assertion
    whose claims bind to the client_id and the respective endpoint."""
    handler = IdJagHandler()
    server = _id_jag_server(
        client_secret=None,
        client_private_key=_PRIVATE_PEM,
        client_private_key_id="kid-1",
    )
    mock_client = AsyncMock()
    mock_client.post.side_effect = [_resp("idjag"), _resp("access")]

    with patch(
        "litellm.proxy._experimental.mcp_server.auth.id_jag.get_async_httpx_client",
        return_value=mock_client,
    ):
        await handler.exchange_token("user-id-token", server)

    for index, endpoint in ((0, _LEG1_ENDPOINT), (1, _LEG2_ENDPOINT)):
        _, data = _leg(mock_client, index)
        assert data["client_assertion_type"] == CLIENT_ASSERTION_TYPE
        assert "client_secret" not in data
        decoded = jwt.decode(
            data["client_assertion"],
            _PUBLIC_PEM,
            algorithms=["RS256"],
            audience=endpoint,
        )
        assert decoded["iss"] == "litellm-client-id"
        assert decoded["sub"] == "litellm-client-id"
        assert decoded["aud"] == endpoint
        assert "exp" in decoded
        assert jwt.get_unverified_header(data["client_assertion"])["kid"] == "kid-1"


@pytest.mark.asyncio
async def test_client_secret_fallback_when_no_private_key():
    """Without a private key, both legs authenticate with client_id/client_secret."""
    handler = IdJagHandler()
    server = _id_jag_server()
    mock_client = AsyncMock()
    mock_client.post.side_effect = [_resp("idjag"), _resp("access")]

    with patch(
        "litellm.proxy._experimental.mcp_server.auth.id_jag.get_async_httpx_client",
        return_value=mock_client,
    ):
        await handler.exchange_token("user-id-token", server)

    for index in (0, 1):
        _, data = _leg(mock_client, index)
        assert data["client_id"] == "litellm-client-id"
        assert data["client_secret"] == "litellm-client-secret"
        assert "client_assertion" not in data


# ── Caching ──


@pytest.mark.asyncio
async def test_cached_skips_both_legs():
    """A second call with the same subject_token serves from cache (2 POSTs total, not 4)."""
    handler = IdJagHandler()
    server = _id_jag_server()
    mock_client = AsyncMock()
    mock_client.post.side_effect = [_resp("idjag"), _resp("cached-access")]

    with patch(
        "litellm.proxy._experimental.mcp_server.auth.id_jag.get_async_httpx_client",
        return_value=mock_client,
    ):
        t1 = await handler.exchange_token("same-id-token", server)
        t2 = await handler.exchange_token("same-id-token", server)

    assert t1 == t2 == "cached-access"
    assert mock_client.post.call_count == 2


@pytest.mark.asyncio
async def test_concurrent_calls_exchange_once():
    """Two concurrent calls for the same subject_token share a single exchange;
    the loser waits on the lock and serves from cache (2 POSTs total, not 4)."""
    handler = IdJagHandler()
    server = _id_jag_server()
    responses = iter([_resp("idjag"), _resp("shared-access")])
    call_count = 0

    async def slow_post(url, data=None):
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.01)
        return next(responses)

    mock_client = AsyncMock()
    mock_client.post = slow_post

    with patch(
        "litellm.proxy._experimental.mcp_server.auth.id_jag.get_async_httpx_client",
        return_value=mock_client,
    ):
        results = await asyncio.gather(
            handler.exchange_token("same-id-token", server),
            handler.exchange_token("same-id-token", server),
        )

    assert results == ["shared-access", "shared-access"]
    assert call_count == 2


@pytest.mark.asyncio
async def test_invalidate_forces_re_exchange():
    """invalidate() drops the cached token so the next call exchanges again."""
    handler = IdJagHandler()
    server = _id_jag_server()
    mock_client = AsyncMock()
    mock_client.post.side_effect = [
        _resp("idjag-1"),
        _resp("access-1"),
        _resp("idjag-2"),
        _resp("access-2"),
    ]

    with patch(
        "litellm.proxy._experimental.mcp_server.auth.id_jag.get_async_httpx_client",
        return_value=mock_client,
    ):
        first = await handler.exchange_token("id-token", server)
        handler.invalidate("id-token", server.server_id)
        second = await handler.exchange_token("id-token", server)

    assert first == "access-1"
    assert second == "access-2"
    assert mock_client.post.call_count == 4


# ── Error handling ──


@pytest.mark.asyncio
async def test_leg1_http_error():
    """A leg-1 IdP error raises ValueError naming leg 1."""
    handler = IdJagHandler()
    server = _id_jag_server()
    error_resp = MagicMock()
    error_resp.status_code = 400
    error_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Bad Request", request=MagicMock(), response=error_resp
    )
    mock_client = AsyncMock()
    mock_client.post.return_value = error_resp

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.auth.id_jag.get_async_httpx_client",
            return_value=mock_client,
        ),
        pytest.raises(ValueError, match="leg 1.*failed with status 400"),
    ):
        await handler.exchange_token("user-id-token", server)


@pytest.mark.asyncio
async def test_leg2_http_error():
    """A leg-2 resource-AS error raises ValueError naming leg 2 (after leg 1 succeeds)."""
    handler = IdJagHandler()
    server = _id_jag_server()
    error_resp = MagicMock()
    error_resp.status_code = 403
    error_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Forbidden", request=MagicMock(), response=error_resp
    )
    mock_client = AsyncMock()
    mock_client.post.side_effect = [_resp("idjag"), error_resp]

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.auth.id_jag.get_async_httpx_client",
            return_value=mock_client,
        ),
        pytest.raises(ValueError, match="leg 2.*failed with status 403"),
    ):
        await handler.exchange_token("user-id-token", server)


@pytest.mark.asyncio
async def test_leg1_missing_access_token():
    """A leg-1 response without access_token raises ValueError."""
    handler = IdJagHandler()
    server = _id_jag_server()
    bad = MagicMock()
    bad.json.return_value = {"token_type": "Bearer"}
    bad.raise_for_status = MagicMock()
    mock_client = AsyncMock()
    mock_client.post.return_value = bad

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.auth.id_jag.get_async_httpx_client",
            return_value=mock_client,
        ),
        pytest.raises(ValueError, match="leg 1.*missing 'access_token'"),
    ):
        await handler.exchange_token("user-id-token", server)


@pytest.mark.asyncio
async def test_missing_endpoint_raises():
    """A missing leg-2 endpoint raises ValueError before any HTTP call."""
    handler = IdJagHandler()
    server = _id_jag_server(id_jag_resource_token_endpoint=None)

    with pytest.raises(
        ValueError, match="token_exchange_endpoint or id_jag_resource_token_endpoint"
    ):
        await handler._do_exchange("user-id-token", server)


@pytest.mark.asyncio
async def test_missing_client_id_raises():
    """A missing client_id raises ValueError before any HTTP call."""
    handler = IdJagHandler()
    server = _id_jag_server(client_id=None)

    with pytest.raises(ValueError, match="missing client_id"):
        await handler._do_exchange("user-id-token", server)


@pytest.mark.asyncio
async def test_missing_client_auth_raises():
    """No private key and no client_secret raises ValueError."""
    handler = IdJagHandler()
    server = _id_jag_server(client_secret=None, client_private_key=None)

    with pytest.raises(ValueError, match="client_private_key or client_secret"):
        await handler._do_exchange("user-id-token", server)


@pytest.mark.asyncio
async def test_none_response_raises():
    """A None response from the HTTP client raises ValueError rather than crashing."""
    handler = IdJagHandler()
    server = _id_jag_server()
    mock_client = AsyncMock()
    mock_client.post.return_value = None

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.auth.id_jag.get_async_httpx_client",
            return_value=mock_client,
        ),
        pytest.raises(ValueError, match="returned no response"),
    ):
        await handler.exchange_token("user-id-token", server)


# ── resolve_mcp_auth integration ──


@pytest.mark.asyncio
async def test_resolve_mcp_auth_routes_to_id_jag():
    """resolve_mcp_auth delegates to the ID-JAG handler when configured with a subject_token."""
    server = _id_jag_server()
    mock_handler = AsyncMock()
    mock_handler.exchange_token.return_value = "idjag-access-token"

    with patch(
        "litellm.proxy._experimental.mcp_server.auth.id_jag.mcp_id_jag_handler",
        mock_handler,
    ):
        result = await resolve_mcp_auth(server, subject_token="user-id-token")

    assert result == "idjag-access-token"
    mock_handler.exchange_token.assert_called_once_with("user-id-token", server)


@pytest.mark.asyncio
async def test_resolve_mcp_auth_id_jag_without_subject_token():
    """Without a subject_token, ID-JAG does not call the handler and returns no token."""
    server = _id_jag_server(authentication_token=None)
    mock_handler = AsyncMock()

    with patch(
        "litellm.proxy._experimental.mcp_server.auth.id_jag.mcp_id_jag_handler",
        mock_handler,
    ):
        result = await resolve_mcp_auth(server, subject_token=None)

    assert result is None
    mock_handler.exchange_token.assert_not_called()


@pytest.mark.asyncio
async def test_resolve_mcp_auth_header_beats_id_jag():
    """An explicit mcp_auth_header takes priority over the ID-JAG flow."""
    server = _id_jag_server()
    result = await resolve_mcp_auth(
        server, mcp_auth_header="Bearer override", subject_token="user-id-token"
    )
    assert result == "Bearer override"


# ── has_id_jag_config ──


def test_has_id_jag_config_true():
    assert _id_jag_server().has_id_jag_config is True


def test_has_id_jag_config_true_with_private_key_only():
    server = _id_jag_server(client_secret=None, client_private_key=_PRIVATE_PEM)
    assert server.has_id_jag_config is True


def test_has_id_jag_config_false_wrong_auth_type():
    assert (
        _id_jag_server(auth_type=MCPAuth.oauth2_token_exchange).has_id_jag_config
        is False
    )


def test_has_id_jag_config_false_missing_leg1_endpoint():
    assert _id_jag_server(token_exchange_endpoint=None).has_id_jag_config is False


def test_has_id_jag_config_false_missing_leg2_endpoint():
    assert (
        _id_jag_server(id_jag_resource_token_endpoint=None).has_id_jag_config is False
    )


def test_has_id_jag_config_false_missing_client_id():
    assert _id_jag_server(client_id=None).has_id_jag_config is False


def test_has_id_jag_config_false_no_client_auth():
    server = _id_jag_server(client_secret=None, client_private_key=None)
    assert server.has_id_jag_config is False


# ── Config / DB loading ──


@pytest.mark.asyncio
async def test_config_loading_id_jag_fields():
    """load_servers_from_config maps the ID-JAG config fields onto MCPServer."""
    manager = MCPServerManager()
    config = {
        "my_idjag_server": {
            "url": "https://mcp.example.com/mcp",
            "transport": "http",
            "auth_type": "oauth2_id_jag",
            "client_id": "my-client",
            "token_exchange_endpoint": _LEG1_ENDPOINT,
            "id_jag_resource_token_endpoint": _LEG2_ENDPOINT,
            "id_jag_resource": "https://mcp.example.com/",
            "audience": "api://my-mcp",
            "client_private_key": _PRIVATE_PEM,
            "client_private_key_id": "kid-1",
            "client_assertion_signing_alg": "RS384",
        }
    }
    await manager.load_servers_from_config(config)

    server = list(manager.config_mcp_servers.values())[0]
    assert server.auth_type == MCPAuth.oauth2_id_jag
    assert server.token_exchange_endpoint == _LEG1_ENDPOINT
    assert server.id_jag_resource_token_endpoint == _LEG2_ENDPOINT
    assert server.id_jag_resource == "https://mcp.example.com/"
    assert server.client_private_key == _PRIVATE_PEM
    assert server.client_private_key_id == "kid-1"
    assert server.client_assertion_signing_alg == "RS384"
    assert server.has_id_jag_config is True


@pytest.mark.asyncio
async def test_database_loading_id_jag_fields_from_credentials():
    """DB-loaded ID-JAG servers retain the ID-JAG credential fields."""
    manager = MCPServerManager()
    db_server = LiteLLM_MCPServerTable(
        server_id="srv-idjag-db",
        server_name="idjag_db_server",
        url="https://mcp.example.com/mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2_id_jag,
        credentials={
            "client_id": "db-client",
            "token_exchange_endpoint": _LEG1_ENDPOINT,
            "id_jag_resource_token_endpoint": _LEG2_ENDPOINT,
            "client_private_key": _PRIVATE_PEM,
            "client_private_key_id": "db-kid",
        },
    )

    server = await manager.build_mcp_server_from_table(
        db_server,
        credentials_are_encrypted=False,
    )

    assert server.auth_type == MCPAuth.oauth2_id_jag
    assert server.client_id == "db-client"
    assert server.id_jag_resource_token_endpoint == _LEG2_ENDPOINT
    assert server.client_private_key == _PRIVATE_PEM
    assert server.client_private_key_id == "db-kid"
    assert server.has_id_jag_config is True
