"""
ID-JAG (Identity Assertion Authorization Grant) handler for MCP servers.

Two-legged egress flow for the case where the MCP's authorization server is a
different server than the user's IdP (draft-ietf-oauth-identity-assertion-authz-grant,
shipped by Okta as "AI agent token exchange"):

Leg 1 (RFC 8693 token exchange): swap the user's id_token for an ID-JAG assertion
at the IdP's org authorization server.
Leg 2 (RFC 7523 JWT-bearer): present that assertion to the MCP's resource
authorization server to obtain the access token used to call the MCP.

The gateway authenticates to both authorization servers with a private-key-JWT
client_assertion (RFC 7523), falling back to client_secret when no key is set.
"""

import asyncio
import hashlib
import time
import uuid
import weakref
from typing import TYPE_CHECKING, Optional

import httpx
import jwt
from pydantic import BaseModel, ValidationError

from litellm._logging import verbose_logger
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.constants import (
    MCP_OAUTH2_TOKEN_CACHE_DEFAULT_TTL,
    MCP_OAUTH2_TOKEN_CACHE_MIN_TTL,
    MCP_OAUTH2_TOKEN_EXPIRY_BUFFER_SECONDS,
    MCP_TOKEN_EXCHANGE_CACHE_MAX_SIZE,
)
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    get_async_httpx_client,
)
from litellm.types.llms.custom_http import httpxSpecialProvider

if TYPE_CHECKING:
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

TOKEN_EXCHANGE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:token-exchange"
JWT_BEARER_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:jwt-bearer"
ID_JAG_REQUESTED_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:id-jag"
DEFAULT_ID_JAG_SUBJECT_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:id_token"
TOKEN_EXCHANGE_SUBJECT_TOKEN_DEFAULT = "urn:ietf:params:oauth:token-type:access_token"
CLIENT_ASSERTION_TYPE = "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"
CLIENT_ASSERTION_LIFETIME_SECONDS = 60


class _OAuthTokenResponse(BaseModel):
    access_token: str
    expires_in: Optional[int] = None


class IdJagHandler:
    """Performs the two-legged ID-JAG egress flow for MCP servers.

    Caches the leg-2 access token keyed by ``hash(subject_token + server_id)`` so
    repeated calls with the same user token skip both authorization-server round-trips.
    """

    def __init__(self) -> None:
        self._cache = InMemoryCache(
            max_size_in_memory=MCP_TOKEN_EXCHANGE_CACHE_MAX_SIZE,
            default_ttl=MCP_OAUTH2_TOKEN_CACHE_DEFAULT_TTL,
        )
        self._locks: weakref.WeakValueDictionary[str, asyncio.Lock] = (
            weakref.WeakValueDictionary()
        )

    def _get_lock(self, cache_key: str) -> asyncio.Lock:
        lock = self._locks.get(cache_key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[cache_key] = lock
        return lock

    @staticmethod
    def _cache_key(subject_token: str, server_id: str) -> str:
        raw = f"{subject_token}:{server_id}"
        return hashlib.sha256(raw.encode()).hexdigest()

    async def exchange_token(
        self,
        subject_token: str,
        server: "MCPServer",
    ) -> str:
        """Run the ID-JAG flow for *subject_token* and return the MCP access token.

        Raises ``ValueError`` on configuration or authorization-server errors.
        """
        cache_key = self._cache_key(subject_token, server.server_id)

        cached = self._cache.get_cache(cache_key)
        if cached is not None:
            return cached

        async with self._get_lock(cache_key):
            cached = self._cache.get_cache(cache_key)
            if cached is not None:
                return cached

            token, ttl = await self._do_exchange(subject_token, server)
            self._cache.set_cache(cache_key, token, ttl=ttl)
            return token

    async def _do_exchange(
        self,
        subject_token: str,
        server: "MCPServer",
    ) -> tuple[str, int]:
        leg1_endpoint = server.token_exchange_endpoint
        leg2_endpoint = server.id_jag_resource_token_endpoint
        if not leg1_endpoint or not leg2_endpoint:
            raise ValueError(
                f"MCP server '{server.server_id}' has auth_type=oauth2_id_jag but is "
                f"missing token_exchange_endpoint or id_jag_resource_token_endpoint"
            )
        client_id = server.client_id
        if not client_id:
            raise ValueError(
                f"MCP server '{server.server_id}' has auth_type=oauth2_id_jag but "
                f"missing client_id"
            )

        client = get_async_httpx_client(llm_provider=httpxSpecialProvider.MCP)

        leg1_data: dict[str, str] = {
            "grant_type": TOKEN_EXCHANGE_GRANT_TYPE,
            "requested_token_type": ID_JAG_REQUESTED_TOKEN_TYPE,
            "subject_token": subject_token,
            "subject_token_type": self._subject_token_type(server),
            **({"audience": server.audience} if server.audience else {}),
            **({"resource": server.id_jag_resource} if server.id_jag_resource else {}),
            **({"scope": " ".join(server.scopes)} if server.scopes else {}),
            **self._client_auth(client_id, server, leg1_endpoint),
        }
        id_jag = (
            await self._request_token(
                client, leg1_endpoint, leg1_data, server, "token exchange (leg 1)"
            )
        ).access_token

        leg2_data: dict[str, str] = {
            "grant_type": JWT_BEARER_GRANT_TYPE,
            "assertion": id_jag,
            **self._client_auth(client_id, server, leg2_endpoint),
        }
        leg2 = await self._request_token(
            client, leg2_endpoint, leg2_data, server, "JWT-bearer (leg 2)"
        )

        expires_in = (
            leg2.expires_in
            if leg2.expires_in is not None
            else MCP_OAUTH2_TOKEN_CACHE_DEFAULT_TTL
        )
        ttl = max(
            expires_in - MCP_OAUTH2_TOKEN_EXPIRY_BUFFER_SECONDS,
            MCP_OAUTH2_TOKEN_CACHE_MIN_TTL,
        )

        verbose_logger.info(
            "ID-JAG exchange succeeded for MCP server %s (expires in %ds)",
            server.server_id,
            expires_in,
        )
        return leg2.access_token, ttl

    @staticmethod
    def _subject_token_type(server: "MCPServer") -> str:
        configured = server.subject_token_type
        if configured and configured != TOKEN_EXCHANGE_SUBJECT_TOKEN_DEFAULT:
            return configured
        return DEFAULT_ID_JAG_SUBJECT_TOKEN_TYPE

    @staticmethod
    def _client_auth(
        client_id: str, server: "MCPServer", audience: str
    ) -> dict[str, str]:
        if server.client_private_key:
            now = int(time.time())
            assertion = jwt.encode(
                {
                    "iss": client_id,
                    "sub": client_id,
                    "aud": audience,
                    "jti": uuid.uuid4().hex,
                    "iat": now,
                    "exp": now + CLIENT_ASSERTION_LIFETIME_SECONDS,
                },
                server.client_private_key,
                algorithm=server.client_assertion_signing_alg,
                headers=(
                    {"kid": server.client_private_key_id}
                    if server.client_private_key_id
                    else None
                ),
            )
            return {
                "client_id": client_id,
                "client_assertion_type": CLIENT_ASSERTION_TYPE,
                "client_assertion": assertion,
            }
        if server.client_secret:
            return {"client_id": client_id, "client_secret": server.client_secret}
        raise ValueError(
            f"MCP server '{server.server_id}' has auth_type=oauth2_id_jag but no "
            f"client_private_key or client_secret configured"
        )

    @staticmethod
    async def _request_token(
        client: AsyncHTTPHandler,
        endpoint: str,
        data: dict[str, str],
        server: "MCPServer",
        leg: str,
    ) -> _OAuthTokenResponse:
        try:
            response = await client.post(endpoint, data=data)
            if response is None:
                raise ValueError(
                    f"ID-JAG {leg} for MCP server '{server.server_id}' returned "
                    f"no response"
                )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            verbose_logger.debug(
                "ID-JAG %s error for MCP server %s (status %d)",
                leg,
                server.server_id,
                exc.response.status_code,
            )
            raise ValueError(
                f"ID-JAG {leg} for MCP server '{server.server_id}' failed with "
                f"status {exc.response.status_code}"
            ) from exc

        try:
            return _OAuthTokenResponse.model_validate(response.json())
        except ValidationError as exc:
            raise ValueError(
                f"ID-JAG {leg} response for MCP server '{server.server_id}' "
                f"missing 'access_token'"
            ) from exc

    def invalidate(self, subject_token: str, server_id: str) -> None:
        cache_key = self._cache_key(subject_token, server_id)
        self._cache.delete_cache(cache_key)


mcp_id_jag_handler = IdJagHandler()
