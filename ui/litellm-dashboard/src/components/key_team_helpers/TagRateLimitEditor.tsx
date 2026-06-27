import { Button, Input, InputNumber } from "antd";
import React from "react";

export interface TagRateLimitEntry {
  // Stable identity for React list keys so deleting a middle row doesn't shift
  // the controlled inputs of the rows below it.
  id: string;
  tag: string;
  rpm_limit: number | null;
  tpm_limit: number | null;
}

let nextRowId = 0;
const newRowId = (): string => `tag-row-${nextRowId++}`;

export interface TagRateLimits {
  tag_rpm_limit: Record<string, number>;
  tag_tpm_limit: Record<string, number>;
}

// Build the two limit maps from editor rows. A tag only enters a map when its
// name is non-empty and that limit cell holds a number, so a row with just an
// RPM value lands in tag_rpm_limit and stays out of tag_tpm_limit.
export const tagRowsToLimits = (rows: TagRateLimitEntry[]): TagRateLimits => {
  const tag_rpm_limit: Record<string, number> = {};
  const tag_tpm_limit: Record<string, number> = {};
  rows.forEach(({ tag, rpm_limit, tpm_limit }) => {
    const name = tag.trim();
    if (!name) return;
    if (typeof rpm_limit === "number") tag_rpm_limit[name] = rpm_limit;
    if (typeof tpm_limit === "number") tag_tpm_limit[name] = tpm_limit;
  });
  return { tag_rpm_limit, tag_tpm_limit };
};

// Coerce an untyped metadata value into a {tag: number} map, dropping anything
// that isn't a numeric entry. Key metadata is loosely typed, so validate here.
const toNumberMap = (raw: unknown): Record<string, number> => {
  if (!raw || typeof raw !== "object") return {};
  const out: Record<string, number> = {};
  Object.entries(raw as Record<string, unknown>).forEach(([tag, limit]) => {
    if (typeof limit === "number") out[tag] = limit;
  });
  return out;
};

// Reconstruct editor rows from the two stored maps (union of their tags).
export const tagLimitsToRows = (tagRpmLimit?: unknown, tagTpmLimit?: unknown): TagRateLimitEntry[] => {
  const rpm = toNumberMap(tagRpmLimit);
  const tpm = toNumberMap(tagTpmLimit);
  const tags = Array.from(new Set([...Object.keys(rpm), ...Object.keys(tpm)]));
  return tags.map((tag) => ({
    id: newRowId(),
    tag,
    rpm_limit: tag in rpm ? rpm[tag] : null,
    tpm_limit: tag in tpm ? tpm[tag] : null,
  }));
};

interface TagRateLimitEditorProps {
  value: TagRateLimitEntry[];
  onChange: (v: TagRateLimitEntry[]) => void;
}

export function TagRateLimitEditor({ value, onChange }: TagRateLimitEditorProps) {
  const addRow = () => {
    onChange([...value, { id: newRowId(), tag: "", rpm_limit: null, tpm_limit: null }]);
  };

  const removeRow = (idx: number) => {
    onChange(value.filter((_, i) => i !== idx));
  };

  const updateRow = (idx: number, field: keyof TagRateLimitEntry, fieldValue: string | number | null) => {
    onChange(value.map((row, i) => (i === idx ? { ...row, [field]: fieldValue } : row)));
  };

  return (
    <div>
      {value.map((row, idx) => (
        <div key={row.id} style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 12 }}>
          <Input
            value={row.tag}
            onChange={(e) => updateRow(idx, "tag", e.target.value)}
            placeholder="Tag (e.g. cell-1)"
            style={{ width: 180 }}
          />
          <InputNumber
            min={0}
            value={row.rpm_limit ?? undefined}
            onChange={(v) => updateRow(idx, "rpm_limit", v ?? null)}
            placeholder="RPM"
            style={{ width: 120 }}
          />
          <InputNumber
            min={0}
            value={row.tpm_limit ?? undefined}
            onChange={(v) => updateRow(idx, "tpm_limit", v ?? null)}
            placeholder="TPM"
            style={{ width: 120 }}
          />
          <Button type="text" danger size="small" onClick={() => removeRow(idx)} style={{ padding: "0 4px" }}>
            ✕
          </Button>
        </div>
      ))}
      <Button
        size="small"
        onClick={(e) => {
          e.preventDefault();
          addRow();
        }}
      >
        + Add Tag Limit
      </Button>
    </div>
  );
}
