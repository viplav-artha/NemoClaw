// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export const WEB_SEARCH_PROVIDERS = [
  "brave",
  "tavily",
  "duckduckgo",
  "parallel-free",
  "firecrawl-free",
  "searxng",
  "ollama",
] as const;

export type WebSearchProvider = (typeof WEB_SEARCH_PROVIDERS)[number];

const WEB_SEARCH_PROVIDER_SET: ReadonlySet<string> = new Set(WEB_SEARCH_PROVIDERS);

export function isWebSearchProvider(value: unknown): value is WebSearchProvider {
  return typeof value === "string" && WEB_SEARCH_PROVIDER_SET.has(value);
}
