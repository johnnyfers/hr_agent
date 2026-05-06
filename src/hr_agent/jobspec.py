"""Client/Job/Field specifications.

A JobSpec fully describes one screening flow: what to ask, in what order,
which answers disqualify, the FAQ that backs the agent, and (for delivery-
type roles) the service-area whitelist. The agent reads from this; nothing
about the role is hard-coded.

Specs live in the database (one row per job). The shape is versioned with
``spec_version`` so we can introduce field types over time without breaking
older rows.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

LangText = dict[str, str]  # { "es": "...", "en": "..." }

FieldType = Literal[
    "bool",
    "string",
    "enum",
    "int",
    "city",
    "experience",
    "date",
]

DisqualifyRule = Literal[
    "false",                 # disqualify when the bool is false
    "true",                  # disqualify when the bool is true (rare but symmetric)
    "out_of_service_area",   # disqualify when a city field is not in the spec's service_areas
    "out_of_set",            # disqualify when an enum/string value is in `disqualify_values`
]


class FieldSpec(BaseModel):
    name: str  # e.g. "has_license"; used as the storage key in ScreeningState.fields
    type: FieldType
    required: bool = True
    disqualify_when: Optional[DisqualifyRule] = None
    disqualify_values: list[str] = Field(default_factory=list)  # for out_of_set

    enum_values: list[str] = Field(default_factory=list)  # required when type == "enum"

    label: LangText  # short human label, e.g. {"es": "Permiso", "en": "Licence"}
    prompt_hint: LangText  # the actual question copy the agent draws from

    # Optional bounds for type=int
    min: Optional[int] = None
    max: Optional[int] = None


class FAQEntry(BaseModel):
    id: str
    tags: list[str]
    text: LangText  # answer, by language


class ServiceAreas(BaseModel):
    """Whitelist of cities, organised by country, plus aliases.

    Empty/unset → no service-area filtering (any city accepted).
    """

    countries: dict[str, list[str]] = Field(default_factory=dict)  # {"ES": ["Madrid", ...], ...}
    aliases: dict[str, str] = Field(default_factory=dict)  # {"CDMX": "Ciudad de México"}


class Client(BaseModel):
    id: str
    name: str


class JobSpec(BaseModel):
    """The contract between the agent and one specific (client, role) pairing."""

    spec_version: int = 1

    client: Client
    job_id: str  # unique, conventional shape: "<client_id>/<role-slug>"
    title: LangText
    description: str = ""
    languages: list[str] = Field(default_factory=lambda: ["es", "en"])
    default_language: str = "es"

    fields: list[FieldSpec]
    service_areas: Optional[ServiceAreas] = None
    faq: list[FAQEntry] = Field(default_factory=list)
    company_blurb: LangText = Field(default_factory=dict)

    tone_notes: str = ""

    def field(self, name: str) -> Optional[FieldSpec]:
        return next((f for f in self.fields if f.name == name), None)

    def has_city_field(self) -> bool:
        return any(f.type == "city" for f in self.fields)
