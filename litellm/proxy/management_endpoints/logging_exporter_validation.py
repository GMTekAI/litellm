"""Validation for admin-owned logging-exporter assignment on key/team/org.

An identity's ``metadata.logging_exporters`` binds it to admin-owned trace
destinations. Every name must be a registered logging credential, and the
caller must hold a role that authorizes the write (proxy admin always; team
admin and org admin in specific contexts). The resolver
(``litellm_pre_call_utils``) trusts this gate at request time.
"""

from typing import Optional

from fastapi import HTTPException, status

import litellm
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth

LOGGING_EXPORTERS_KEY = "logging_exporters"


def is_admin_gated_credential_info(credential_info: Optional[dict]) -> bool:
    """Whether a credential write must be proxy-admin only.

    True when the credential is a logging destination or carries an ``access`` grant,
    since both control where other tenants' traces are exported.
    """
    if not isinstance(credential_info, dict):
        return False
    return (
        credential_info.get("credential_type") == "logging"
        or "access" in credential_info
    )


def validate_credential_access(credential_info: Optional[dict]) -> None:
    """Validate ``credential_info.access`` shape when the write sets one.

    No-op when ``access`` is absent. Otherwise it must be an object whose ``global`` (if
    present) is a bool and whose ``teams``/``orgs`` (if present) are lists of strings.
    Per-key access is intentionally unsupported on a destination.
    """
    if not isinstance(credential_info, dict) or "access" not in credential_info:
        return
    access = credential_info["access"]
    if not isinstance(access, dict):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "credential_info.access must be an object"},
        )
    if "global" in access and not isinstance(access["global"], bool):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "access.global must be a boolean"},
        )
    for field in ("teams", "orgs"):
        bucket = access.get(field)
        if bucket is not None and not (
            isinstance(bucket, list) and all(isinstance(item, str) for item in bucket)
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": f"access.{field} must be a list of strings"},
            )


def _logging_credential_names() -> set:
    return {
        credential.credential_name
        for credential in litellm.credential_list
        if (credential.credential_info or {}).get("credential_type") == "logging"
    }


def _validate_exporters_shape_and_names(exporters: object) -> None:
    """Common shape + registry check shared by every entry point."""
    if not isinstance(exporters, list):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "logging_exporters must be a list of credential names"},
        )
    known = _logging_credential_names()
    unknown = [name for name in exporters if name not in known]
    if unknown:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": (
                    f"Unknown or non-logging credential(s): {unknown}. Register them "
                    "as logging credentials before assigning."
                )
            },
        )


def _exporter_value_changes(
    requested_metadata: Optional[dict],
    existing_metadata: Optional[dict],
) -> bool:
    """True if the effective ``metadata.logging_exporters`` value would change.

    An update endpoint that REPLACES stored metadata with ``requested_metadata``
    will drop ``logging_exporters`` when the new payload omits it. So a write
    requires authorization whenever:

    - the new metadata sets ``logging_exporters`` (the previously-handled case), OR
    - the new metadata is provided but omits ``logging_exporters`` while the
      stored metadata had one (removal-via-omission, Veria F4).

    Returns False when stored and requested values match exactly, or when the
    update doesn't touch metadata at all.
    """
    if not isinstance(requested_metadata, dict):
        return False
    new_has = LOGGING_EXPORTERS_KEY in requested_metadata
    existing = (
        existing_metadata.get(LOGGING_EXPORTERS_KEY)
        if isinstance(existing_metadata, dict)
        else None
    )
    existing_has = existing is not None
    if not new_has and not existing_has:
        return False
    if new_has and not existing_has:
        return True
    if not new_has and existing_has:
        return True
    return requested_metadata.get(LOGGING_EXPORTERS_KEY) != existing


def validate_logging_exporter_assignment(
    metadata: Optional[dict],
    user_api_key_dict: UserAPIKeyAuth,
    *,
    caller_is_team_admin: bool = False,
    caller_is_org_admin: bool = False,
    existing_metadata: Optional[dict] = None,
) -> None:
    """Validate a ``metadata.logging_exporters`` write on key / team / org endpoints.

    No-op when the update does not change the effective ``logging_exporters``
    value. Proxy admins always pass. Caller-provided flags widen the allow-list
    per endpoint:

    - ``/team/update``: pass ``caller_is_org_admin`` from the loaded team's org.
    - ``/key/generate``/``/key/update``: pass both flags from the key's team.
    - ``/team/new``/``/organization/*``: pass neither (proxy-admin only).

    Update paths replace stored metadata wholesale, so a caller can drop an
    admin-assigned exporter by sending ``metadata`` without
    ``logging_exporters``. Pass ``existing_metadata`` from the loaded row so
    removal-via-omission is gated too (Veria F4). On create paths the existing
    value is implicitly ``None`` and the validator behaves as before.

    Every exporter name (when present) must resolve to a registered logging credential.
    """
    if not _exporter_value_changes(metadata, existing_metadata):
        return
    is_proxy_admin = user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN
    if not (is_proxy_admin or caller_is_team_admin or caller_is_org_admin):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": (
                    "Only the proxy admin, a team admin of this team, or an "
                    "org admin of this team's organization can assign logging "
                    "exporters"
                )
            },
        )
    requested = (
        metadata.get(LOGGING_EXPORTERS_KEY) if isinstance(metadata, dict) else None
    )
    if requested is not None:
        _validate_exporters_shape_and_names(requested)
