// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { isWebSearchProvider, type WebSearchProvider } from "./web-search/provider";

export {
  isWebSearchProvider,
  WEB_SEARCH_PROVIDERS,
  type WebSearchProvider,
} from "./web-search/provider";

export interface WebSearchConfig {
  fetchEnabled: boolean;
  /**
   * Optional only for compatibility with sessions and callers created before
   * provider selection existed. Every persistence and runtime boundary
   * normalizes a missing provider to Brave.
   */
  provider?: WebSearchProvider;
}

export const DEFAULT_WEB_SEARCH_PROVIDER: WebSearchProvider = "brave";
export const WEB_SEARCH_PROVIDER_ENV = "NEMOCLAW_WEB_SEARCH_PROVIDER";
export const BRAVE_API_KEY_ENV = "BRAVE_API_KEY";
export const TAVILY_API_KEY_ENV = "TAVILY_API_KEY";
/**
 * Not a secret — the address of the user's own self-hosted SearXNG instance.
 * Reuses the credential-env plumbing (prompt/store/inject) because the
 * mechanics (ask for a string, persist it, inject via an OpenShell-resolved
 * env var, validate before accepting) are identical to a real credential's;
 * only the prompt copy and validation shape differ.
 */
export const SEARXNG_BASE_URL_ENV = "SEARXNG_BASE_URL";
/**
 * Optional — a locally signed-in Ollama instance needs no key at all; only
 * Ollama's hosted remote search (https://ollama.com) needs this.
 */
export const OLLAMA_API_KEY_ENV = "OLLAMA_API_KEY";

export type ExplicitWebSearchProviderSelection =
  | { specified: false; provider: null }
  | { specified: true; provider: WebSearchProvider | null };

export function parseExplicitWebSearchProvider(
  value: string | null | undefined,
): ExplicitWebSearchProviderSelection {
  const normalized = (value ?? "").trim().toLowerCase();
  if (!normalized) return { specified: false, provider: null };
  if (isWebSearchProvider(normalized)) return { specified: true, provider: normalized };
  if (["none", "off", "disabled", "no", "0"].includes(normalized)) {
    return { specified: true, provider: null };
  }
  throw new Error(
    `Unsupported ${WEB_SEARCH_PROVIDER_ENV}: ${value}. Valid values: brave, tavily, duckduckgo, ` +
      "parallel-free, firecrawl-free, searxng, ollama, none.",
  );
}

export function normalizeWebSearchProvider(value: unknown): WebSearchProvider {
  return isWebSearchProvider(value) ? value : DEFAULT_WEB_SEARCH_PROVIDER;
}

export function webSearchProviderForConfig(
  config: Pick<WebSearchConfig, "provider"> | null | undefined,
): WebSearchProvider {
  return normalizeWebSearchProvider(config?.provider);
}

/**
 * DuckDuckGo, Parallel Search (Free), and Firecrawl Search (Free) are keyless
 * providers (no OpenShell credential provider, no process env var) — callers
 * must handle the `null` case instead of assuming every provider brokers
 * exactly one API key. SearXNG returns a non-secret required config value
 * (the user's own instance URL) through the same slot. Ollama returns a
 * credential env that is optional, not mandatory — see
 * isWebSearchCredentialRequired().
 */
export function webSearchEnvFor(provider: WebSearchProvider): string | null {
  switch (provider) {
    case "duckduckgo":
    case "parallel-free":
    case "firecrawl-free":
      return null;
    case "searxng":
      return SEARXNG_BASE_URL_ENV;
    case "ollama":
      return OLLAMA_API_KEY_ENV;
    case "tavily":
      return TAVILY_API_KEY_ENV;
    default:
      return BRAVE_API_KEY_ENV;
  }
}

/**
 * False for a provider whose env slot (per webSearchEnvFor) is either absent
 * (no credential exists at all) or present-but-optional (Ollama: a locally
 * signed-in instance works with zero config). True for a provider that
 * cannot function without its env value populated (Brave, Tavily, and
 * SearXNG's required instance URL).
 */
export function isWebSearchCredentialRequired(provider: WebSearchProvider): boolean {
  if (provider === "ollama") return false;
  return webSearchEnvFor(provider) !== null;
}

export function webSearchLabelFor(provider: WebSearchProvider): string {
  switch (provider) {
    case "tavily":
      return "Tavily Search";
    case "duckduckgo":
      return "DuckDuckGo Search";
    case "parallel-free":
      return "Parallel Search (Free)";
    case "firecrawl-free":
      return "Firecrawl Search (Free)";
    case "searxng":
      return "SearXNG Search";
    case "ollama":
      return "Ollama Web Search";
    default:
      return "Brave Search";
  }
}

export function webSearchProviderForEnvKey(envKey: string): WebSearchProvider | null {
  if (envKey === BRAVE_API_KEY_ENV) return "brave";
  if (envKey === TAVILY_API_KEY_ENV) return "tavily";
  if (envKey === SEARXNG_BASE_URL_ENV) return "searxng";
  if (envKey === OLLAMA_API_KEY_ENV) return "ollama";
  return null;
}

export function isWebSearchEnabled(
  config: Pick<WebSearchConfig, "fetchEnabled"> | null | undefined,
): boolean {
  return config?.fetchEnabled === true;
}

export function normalizeWebSearchConfig(
  config: Partial<WebSearchConfig> | null | undefined,
): WebSearchConfig | null {
  if (!isWebSearchEnabled(config as WebSearchConfig | null | undefined)) return null;
  const provider =
    config?.provider === undefined
      ? DEFAULT_WEB_SEARCH_PROVIDER
      : isWebSearchProvider(config.provider)
        ? config.provider
        : null;
  if (!provider) return null;
  return {
    fetchEnabled: true,
    provider,
  };
}

export function webSearchConfigsEqual(
  left: Partial<WebSearchConfig> | null | undefined,
  right: Partial<WebSearchConfig> | null | undefined,
): boolean {
  const normalizedLeft = normalizeWebSearchConfig(left);
  const normalizedRight = normalizeWebSearchConfig(right);
  if (!normalizedLeft || !normalizedRight) return normalizedLeft === normalizedRight;
  return normalizedLeft.provider === normalizedRight.provider;
}
