"""Dataset license acknowledgement gates."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetLicense:
    name: str
    requires_acknowledgement: bool
    commercial_use: bool | None = None

    @property
    def is_unknown(self) -> bool:
        return self.name.startswith("UNKNOWN")


def require_license_acknowledgement(
    dataset: str,
    license_info: DatasetLicense,
    *,
    accept: bool,
    acknowledge_unknown: bool,
) -> None:
    accepted = accept or os.environ.get("CFMUSIC_ACCEPT_LICENSES") == "1"
    unknown_accepted = acknowledge_unknown or os.environ.get("CFMUSIC_ACK_UNKNOWN_LICENSE") == "1"
    if license_info.requires_acknowledgement and not accepted:
        raise PermissionError(
            f"{dataset} is licensed as {license_info.name}. Set license.accept=true after "
            "reviewing docs/licenses.md. No data was downloaded."
        )
    if license_info.is_unknown and not unknown_accepted:
        raise PermissionError(
            f"{dataset} has an unverified license. Set license.acknowledge_unknown=true or "
            "CFMUSIC_ACK_UNKNOWN_LICENSE=1 after contacting the dataset authors."
        )
