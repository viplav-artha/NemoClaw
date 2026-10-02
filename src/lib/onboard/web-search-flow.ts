// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { createCurlAuthConfig } from "../adapters/http/auth-config";
import type { OpenShellSandboxBufferedCommandExecutor } from "../adapters/openshell/sandbox-command";
import type { CurlProbeResult } from "../adapters/http/probe";
import { runCurlProbe } from "../adapters/http/probe";
import type { AgentDefinition } from "../agent/defs";
import { getCredential, normalizeCredentialValue, saveCredential } from "../credentials/store";
import {
  BRAVE_API_KEY_ENV,
  isWebSearchCredentialRequired,
  normalizeWebSearchConfig,
  OLLAMA_API_KEY_ENV,
  parseExplicitWebSearchProvider,
  SEARXNG_BASE_URL_ENV,
  TAVILY_API_KEY_ENV,
  WEB_SEARCH_PROVIDER_ENV,
  WEB_SEARCH_PROVIDERS,
  type WebSearchConfig,
  type WebSearchProvider,
  webSearchEnvFor,
  webSearchLabelFor,
  webSearchProviderForConfig,
} from "../inference/web-search";
import { ROOT } from "../runner";
import { classifyValidationFailure } from "../validation";
import { getTransportRecoveryMessage } from "../validation-recovery";
import {
  BACK_TO_SELECTION,
  type BackToSelection,
  isBackToSelection,
} from "./credential-navigation";
import { exitOnboardFromPrompt } from "./prompt-helpers";
import type { ValidationFailureLike } from "./types";
import { agentSupportsWebSearch, agentSupportsWebSearchProvider } from "./web-search-support";
import { verifyWebSearchInsideSandbox as verifyWebSearchInsideSandboxWithDeps } from "./web-search-verify";

const BRAVE_SEARCH_HELP_URL = "https://brave.com/search/api/";
const TAVILY_SEARCH_HELP_URL = "https://app.tavily.com/home";
const DUCKDUCKGO_SEARCH_HELP_URL = "https://docs.openclaw.ai/tools/web";
const PARALLEL_FREE_SEARCH_HELP_URL = "https://docs.openclaw.ai/tools/parallel-search";
const FIRECRAWL_FREE_SEARCH_HELP_URL = "https://docs.openclaw.ai/tools/firecrawl";
const SEARXNG_SEARCH_HELP_URL = "https://docs.openclaw.ai/tools/searxng-search";
const OLLAMA_SEARCH_HELP_URL = "https://docs.openclaw.ai/tools/web";
const WEB_SEARCH_VALIDATION_TIMING_ARGS = ["--connect-timeout", "10", "--max-time", "15"] as const;
// Keyless providers (DuckDuckGo, Parallel Search Free, Firecrawl Search Free)
// never reach validateWebSearchApiKey's header-auth probe (no credential to
// validate), but the Record must stay total over WebSearchProvider for
// type-checking. SearXNG and Ollama use their own dedicated probes below
// instead of this generic header-auth shape.
const CURL_CONFIG_PREFIX: Record<WebSearchProvider, string> = {
  brave: "nemoclaw-brave-probe",
  tavily: "nemoclaw-tavily-probe",
  duckduckgo: "nemoclaw-duckduckgo-probe",
  "parallel-free": "nemoclaw-parallel-free-probe",
  "firecrawl-free": "nemoclaw-firecrawl-free-probe",
  searxng: "nemoclaw-searxng-probe",
  ollama: "nemoclaw-ollama-probe",
};

type WebSearchProviderSpec = {
  provider: WebSearchProvider;
  // null marks a keyless provider: no OpenShell credential provider, no
  // process env var, nothing to prompt for or validate.
  envKey: string | null;
  label: string;
  helpUrl: string;
  /** True for SearXNG: the env slot holds a plain instance URL, not a secret. */
  isUrlConfig?: boolean;
};

const WEB_SEARCH_PROVIDER_SPECS: Record<WebSearchProvider, WebSearchProviderSpec> = {
  brave: {
    provider: "brave",
    envKey: BRAVE_API_KEY_ENV,
    label: webSearchLabelFor("brave"),
    helpUrl: BRAVE_SEARCH_HELP_URL,
  },
  tavily: {
    provider: "tavily",
    envKey: TAVILY_API_KEY_ENV,
    label: webSearchLabelFor("tavily"),
    helpUrl: TAVILY_SEARCH_HELP_URL,
  },
  duckduckgo: {
    provider: "duckduckgo",
    envKey: null,
    label: webSearchLabelFor("duckduckgo"),
    helpUrl: DUCKDUCKGO_SEARCH_HELP_URL,
  },
  "parallel-free": {
    provider: "parallel-free",
    envKey: null,
    label: webSearchLabelFor("parallel-free"),
    helpUrl: PARALLEL_FREE_SEARCH_HELP_URL,
  },
  "firecrawl-free": {
    provider: "firecrawl-free",
    envKey: null,
    label: webSearchLabelFor("firecrawl-free"),
    helpUrl: FIRECRAWL_FREE_SEARCH_HELP_URL,
  },
  searxng: {
    provider: "searxng",
    envKey: SEARXNG_BASE_URL_ENV,
    label: webSearchLabelFor("searxng"),
    helpUrl: SEARXNG_SEARCH_HELP_URL,
    isUrlConfig: true,
  },
  ollama: {
    provider: "ollama",
    envKey: OLLAMA_API_KEY_ENV,
    label: webSearchLabelFor("ollama"),
    helpUrl: OLLAMA_SEARCH_HELP_URL,
  },
};

/** True when the provider brokers no OpenShell credential at all. */
function isKeylessWebSearchProvider(provider: WebSearchProvider): boolean {
  return webSearchEnvFor(provider) === null;
}

/** True only for Ollama: its env slot is optional, not mandatory. */
function isOptionalCredentialWebSearchProvider(provider: WebSearchProvider): boolean {
  return !isKeylessWebSearchProvider(provider) && !isWebSearchCredentialRequired(provider);
}

/** True only for SearXNG: the env slot is a plain instance URL, not a secret. */
function isUrlConfigWebSearchProvider(provider: WebSearchProvider): boolean {
  return WEB_SEARCH_PROVIDER_SPECS[provider].isUrlConfig === true;
}

export interface WebSearchFlowDeps {
  prompt(question: string, options?: { secret?: boolean }): Promise<string>;
  note(message: string): void;
  isNonInteractive(): boolean;
  cliName(): string;
  commandExecutor: OpenShellSandboxBufferedCommandExecutor;
  env?: NodeJS.ProcessEnv;
  getCredential?: (envKey: string) => string | null;
  saveCredential?: (envKey: string, value: string) => void;
}

export interface WebSearchFlowHelpers {
  webSearchProviderForConfig(config: WebSearchConfig): WebSearchProvider;
  validateWebSearchApiKey(provider: WebSearchProvider, apiKey: string): CurlProbeResult;
  validateBraveSearchApiKey(apiKey: string): CurlProbeResult;
  validateTavilySearchApiKey(apiKey: string): CurlProbeResult;
  promptWebSearchRecovery(
    provider: WebSearchProvider,
    validation: ValidationFailureLike,
  ): Promise<"retry" | "skip">;
  promptBraveSearchRecovery(validation: ValidationFailureLike): Promise<"retry" | "skip">;
  promptWebSearchApiKey(provider: WebSearchProvider): Promise<string | BackToSelection>;
  promptBraveSearchApiKey(): Promise<string | BackToSelection>;
  promptWebSearchProvider(
    providers?: readonly WebSearchProvider[],
  ): Promise<WebSearchProvider | null>;
  resolveNonInteractiveWebSearchProvider(): WebSearchProvider | null;
  ensureValidatedWebSearchCredential(
    providerOrConfig: WebSearchProvider | WebSearchConfig,
    nonInteractive?: boolean,
  ): Promise<string | BackToSelection | null>;
  ensureValidatedBraveSearchCredential(
    nonInteractiveOrConfig?: boolean | WebSearchConfig,
  ): Promise<string | BackToSelection | null>;
  configureWebSearch(
    existingConfig?: WebSearchConfig | null,
    agent?: AgentDefinition | null,
    dockerfilePathOverride?: string | null,
  ): Promise<WebSearchConfig | null>;
  verifyWebSearchInsideSandbox(
    sandboxName: string,
    agent: AgentDefinition | null | undefined,
    provider: WebSearchProvider,
  ): Promise<boolean>;
}

export function createWebSearchFlowHelpers(deps: WebSearchFlowDeps): WebSearchFlowHelpers {
  const env = deps.env ?? process.env;
  const readCredential = deps.getCredential ?? getCredential;
  const persistCredential = deps.saveCredential ?? saveCredential;

  function providerSpec(provider: WebSearchProvider): WebSearchProviderSpec {
    return WEB_SEARCH_PROVIDER_SPECS[provider];
  }

  function curlConfigHeaders(provider: WebSearchProvider, apiKey: string): string[] {
    const authHeader =
      provider === "tavily" ? `Authorization: Bearer ${apiKey}` : `X-Subscription-Token: ${apiKey}`;
    return [
      "Accept: application/json",
      ...(provider === "brave" ? ["Accept-Encoding: gzip"] : ["Content-Type: application/json"]),
      authHeader,
    ];
  }

  function validationArgs(provider: WebSearchProvider, authArgs: readonly string[]): string[] {
    if (provider === "tavily") {
      return [
        "-sS",
        ...WEB_SEARCH_VALIDATION_TIMING_ARGS,
        "--compressed",
        ...authArgs,
        "-X",
        "POST",
        "--data-raw",
        JSON.stringify({ query: "ping", max_results: 1 }),
        "https://api.tavily.com/search",
      ];
    }
    return [
      "-sS",
      ...WEB_SEARCH_VALIDATION_TIMING_ARGS,
      "--compressed",
      ...authArgs,
      "--get",
      "--data-urlencode",
      "q=ping",
      "--data-urlencode",
      "count=1",
      "https://api.search.brave.com/res/v1/web/search",
    ];
  }

  function invalidApiKey(provider: WebSearchProvider, message: string): CurlProbeResult {
    return {
      ok: false,
      httpStatus: 0,
      curlStatus: 0,
      body: "",
      stderr: "",
      message: `${providerSpec(provider).label} API key ${message}`,
    };
  }

  // SearXNG's env slot holds the user's own instance URL, not a secret — no
  // Authorization header is sent, just a plain reachability GET against the
  // instance's own /search endpoint with format=json (the same path the real
  // @openclaw/searxng-plugin requests).
  function validateSearxngInstanceUrl(baseUrl: string): CurlProbeResult {
    let parsed: URL;
    try {
      parsed = new URL(baseUrl);
    } catch {
      return invalidApiKey("searxng", "must be a valid URL, e.g. http://localhost:8888.");
    }
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
      return invalidApiKey("searxng", "must use http:// or https://.");
    }
    const probeUrl = new URL("/search", parsed);
    probeUrl.searchParams.set("q", "ping");
    probeUrl.searchParams.set("format", "json");
    return runCurlProbe([
      "-sS",
      ...WEB_SEARCH_VALIDATION_TIMING_ARGS,
      probeUrl.toString(),
    ]);
  }

  // Ollama's remote hosted search (https://ollama.com) is optional and its
  // exact authenticated endpoint shape is not documented publicly — this is
  // a best-effort connectivity check only (any response, including an auth
  // rejection, counts as "reachable"); it does not confirm the key itself is
  // valid the way the Brave/Tavily probes do.
  function validateOllamaApiKey(apiKey: string): CurlProbeResult {
    if (/[\r\n]/.test(apiKey) || apiKey.includes("\0")) {
      return invalidApiKey("ollama", "must not contain control characters.");
    }
    const authConfig = createCurlAuthConfig(
      [{ kind: "header", value: `Authorization: Bearer ${apiKey}` }],
      { prefix: CURL_CONFIG_PREFIX.ollama },
    );
    try {
      return runCurlProbe(
        ["-sS", ...WEB_SEARCH_VALIDATION_TIMING_ARGS, ...authConfig.args, "https://ollama.com"],
        { trustedConfigFiles: authConfig.trustedConfigFiles },
      );
    } finally {
      authConfig.cleanup();
    }
  }

  function validateWebSearchApiKey(provider: WebSearchProvider, apiKey: string): CurlProbeResult {
    if (isKeylessWebSearchProvider(provider)) {
      throw new Error(`${providerSpec(provider).label} does not use an API key; nothing to validate.`);
    }
    if (isUrlConfigWebSearchProvider(provider)) {
      return validateSearxngInstanceUrl(apiKey);
    }
    if (provider === "ollama") {
      return validateOllamaApiKey(apiKey);
    }
    if (/[\r\n]/.test(apiKey)) {
      return invalidApiKey(provider, "must not contain line breaks.");
    }
    if (apiKey.includes("\0")) {
      return invalidApiKey(provider, "must not contain NUL bytes.");
    }
    const authConfig = createCurlAuthConfig(
      curlConfigHeaders(provider, apiKey).map((value) => ({ kind: "header", value })),
      { prefix: CURL_CONFIG_PREFIX[provider] },
    );
    try {
      return runCurlProbe(validationArgs(provider, authConfig.args), {
        trustedConfigFiles: authConfig.trustedConfigFiles,
      });
    } finally {
      authConfig.cleanup();
    }
  }

  function validateBraveSearchApiKey(apiKey: string): CurlProbeResult {
    return validateWebSearchApiKey("brave", apiKey);
  }

  function validateTavilySearchApiKey(apiKey: string): CurlProbeResult {
    return validateWebSearchApiKey("tavily", apiKey);
  }

  async function promptWebSearchRecovery(
    provider: WebSearchProvider,
    validation: ValidationFailureLike,
  ): Promise<"retry" | "skip"> {
    const spec = providerSpec(provider);
    const recovery = classifyValidationFailure(validation);

    if (recovery.kind === "credential") {
      console.log(`  ${spec.label} rejected that API key.`);
    } else if (recovery.kind === "transport") {
      console.log(getTransportRecoveryMessage(validation));
    } else {
      console.log(`  ${spec.label} validation did not succeed.`);
    }

    const answer = (await deps.prompt("  Type 'retry', 'skip', or 'exit' [retry]: "))
      .trim()
      .toLowerCase();
    if (answer === "skip") return "skip";
    if (answer === "exit" || answer === "quit") exitOnboardFromPrompt();
    return "retry";
  }

  function promptBraveSearchRecovery(validation: ValidationFailureLike): Promise<"retry" | "skip"> {
    return promptWebSearchRecovery("brave", validation);
  }

  async function promptWebSearchApiKey(
    provider: WebSearchProvider,
  ): Promise<string | BackToSelection> {
    const spec = providerSpec(provider);
    const fieldLabel = spec.isUrlConfig ? "instance URL" : "API key";
    const optional = isOptionalCredentialWebSearchProvider(provider);
    console.log("");
    if (spec.isUrlConfig) {
      console.log(`  Enter your self-hosted ${spec.label} instance URL, e.g. http://localhost:8888`);
      console.log(`  Docs: ${spec.helpUrl}`);
    } else {
      console.log(`  Get your ${spec.label} API key from: ${spec.helpUrl}`);
      if (optional) {
        console.log("  Leave this blank to use a locally signed-in Ollama instance with no key.");
      }
    }
    console.log("");

    while (true) {
      const value = await deps.prompt(`  ${spec.label} ${fieldLabel}: `, {
        secret: !spec.isUrlConfig,
      });
      const intent = normalizeCredentialValue(value).toLowerCase();
      if (intent === "back") return BACK_TO_SELECTION;
      if (intent === "exit" || intent === "quit") exitOnboardFromPrompt();
      if (intent === "?" || intent === "help") {
        console.log("  Type back to choose again, or exit to quit.");
        continue;
      }
      const key = normalizeCredentialValue(value);
      if (!key) {
        if (optional) return "";
        // Empty input used to loop with no visible escape, leaving Ctrl+C as
        // the only way out (#6025). Surface the existing back/exit options so
        // the user can skip web search instead of being stuck.
        console.error(
          `  ${spec.label} ${fieldLabel} is required. Type back to choose a different option, or exit to quit.`,
        );
        continue;
      }
      return key;
    }
  }

  function promptBraveSearchApiKey(): Promise<string | BackToSelection> {
    return promptWebSearchApiKey("brave");
  }

  // No OpenShell credential exists for a keyless provider (DuckDuckGo), so
  // there is nothing saved to read back.
  function configuredCredential(provider: WebSearchProvider): string {
    const envKey = webSearchEnvFor(provider);
    if (!envKey) return "";
    return readCredential(envKey) || normalizeCredentialValue(env[envKey]);
  }

  function stageValidatedCredential(provider: WebSearchProvider, apiKey: string): void {
    const envKey = webSearchEnvFor(provider);
    if (!envKey) return;
    persistCredential(envKey, apiKey);
    env[envKey] = apiKey;
  }

  // Sentinel used only as a truthy in-memory success signal for a keyless
  // provider. It is never persisted (stageValidatedCredential is a no-op for
  // keyless providers) and never written to any credential store or env var.
  const KEYLESS_PROVIDER_SENTINEL = "(no key needed)";

  async function ensureValidatedWebSearchCredential(
    providerOrConfig: WebSearchProvider | WebSearchConfig,
    nonInteractive = deps.isNonInteractive(),
  ): Promise<string | BackToSelection | null> {
    const provider =
      typeof providerOrConfig === "string"
        ? providerOrConfig
        : webSearchProviderForConfig(providerOrConfig);
    if (isKeylessWebSearchProvider(provider)) return KEYLESS_PROVIDER_SENTINEL;
    const spec = providerSpec(provider);
    const optional = isOptionalCredentialWebSearchProvider(provider);
    // Non-null: the keyless (envKey === null) case already returned above.
    const envKey = spec.envKey as string;
    const savedApiKey = readCredential(envKey);
    let apiKey = savedApiKey || normalizeCredentialValue(env[envKey]);
    let usingSavedKey = Boolean(savedApiKey);

    while (true) {
      if (!apiKey) {
        if (nonInteractive) {
          if (optional) return KEYLESS_PROVIDER_SENTINEL;
          throw new Error(
            `${spec.label} requires ${spec.envKey} or a saved ${spec.label} credential in non-interactive mode.`,
          );
        }
        const promptedApiKey = await promptWebSearchApiKey(provider);
        if (isBackToSelection(promptedApiKey)) return promptedApiKey;
        if (!promptedApiKey && optional) return KEYLESS_PROVIDER_SENTINEL;
        apiKey = promptedApiKey;
        usingSavedKey = false;
      }

      const validation = validateWebSearchApiKey(provider, apiKey);
      if (validation.ok) {
        stageValidatedCredential(provider, apiKey);
        return apiKey;
      }

      const prefix = usingSavedKey
        ? `  Saved ${spec.label} API key validation failed.`
        : `  ${spec.label} API key validation failed.`;
      console.error(prefix);
      if (validation.message) console.error(`  ${validation.message}`);

      if (nonInteractive) {
        throw new Error(
          validation.message || `${spec.label} API key validation failed in non-interactive mode.`,
        );
      }

      const action = await promptWebSearchRecovery(provider, validation);
      if (action === "skip") {
        console.log(`  Skipping ${spec.label} setup.`);
        console.log("");
        return null;
      }
      apiKey = "";
      usingSavedKey = false;
    }
  }

  function ensureValidatedBraveSearchCredential(
    nonInteractiveOrConfig: boolean | WebSearchConfig = deps.isNonInteractive(),
  ): Promise<string | BackToSelection | null> {
    if (typeof nonInteractiveOrConfig === "boolean") {
      return ensureValidatedWebSearchCredential("brave", nonInteractiveOrConfig);
    }
    return ensureValidatedWebSearchCredential(nonInteractiveOrConfig);
  }

  function resolveNonInteractiveWebSearchProvider(): WebSearchProvider | null {
    const explicit = parseExplicitWebSearchProvider(env[WEB_SEARCH_PROVIDER_ENV]);
    if (explicit.specified) return explicit.provider;

    // Preserve the historical implicit behavior: Brave wins when both keys
    // exist. Tavily is auto-selected only when it is the sole configured key.
    if (configuredCredential("brave")) return "brave";
    if (configuredCredential("tavily")) return "tavily";
    return null;
  }

  async function promptWebSearchProvider(
    providers: readonly WebSearchProvider[] = WEB_SEARCH_PROVIDERS,
  ): Promise<WebSearchProvider | null> {
    console.log("");
    console.log("  Enable web search for your agent?");
    console.log("    [1] No web search (default)");
    providers.forEach((provider, index) => {
      console.log(`    [${index + 2}] ${providerSpec(provider).label}`);
    });
    while (true) {
      const maxChoice = providers.length + 1;
      const raw = (await deps.prompt(`  Choose [1-${maxChoice}]: `)).trim().toLowerCase();
      if (raw === "" || raw === "1" || raw === "n" || raw === "no") return null;
      const namedProvider = providers.find((provider) => raw === provider);
      if (namedProvider) return namedProvider;
      const selectedIndex = /^\d+$/.test(raw) ? Number(raw) - 2 : -1;
      if (selectedIndex >= 0 && selectedIndex < providers.length) {
        return providers[selectedIndex];
      }
      // Preserve the former yes/no behavior by selecting the first supported
      // provider. OpenClaw keeps Brave first; Hermes exposes only Tavily.
      if ((raw === "y" || raw === "yes") && providers.length > 0) return providers[0];
      if (raw === "exit" || raw === "quit") exitOnboardFromPrompt();
      console.log(`  Enter a number from 1 to ${maxChoice}.`);
    }
  }

  function providerIsSupported(
    provider: WebSearchProvider,
    agent: AgentDefinition | null,
    dockerfilePathOverride: string | null,
  ): boolean {
    return agentSupportsWebSearchProvider(agent, provider, dockerfilePathOverride, ROOT);
  }

  function providerSupported(
    provider: WebSearchProvider,
    agent: AgentDefinition | null,
    dockerfilePathOverride: string | null,
  ): boolean {
    if (providerIsSupported(provider, agent, dockerfilePathOverride)) return true;
    deps.note(
      `  ${providerSpec(provider).label} is not supported by ${agent?.displayName ?? "this sandbox image"}. Skipping.`,
    );
    return false;
  }

  async function configureNonInteractiveWebSearch(
    existingConfig: WebSearchConfig | null,
    agent: AgentDefinition | null,
    dockerfilePathOverride: string | null,
  ): Promise<WebSearchConfig | null> {
    const explicit = parseExplicitWebSearchProvider(env[WEB_SEARCH_PROVIDER_ENV]);
    if (explicit.specified && !explicit.provider) return null;

    let provider =
      explicit.provider ??
      (existingConfig ? webSearchProviderForConfig(existingConfig) : null) ??
      resolveNonInteractiveWebSearchProvider();

    // Implicit detection keeps Brave-first precedence among providers the
    // selected agent actually supports. Thus OpenClaw remains backward
    // compatible, while Hermes can use a configured Tavily key even when an
    // unrelated Brave key is also present in the host credential store.
    if (!explicit.specified && !existingConfig) {
      provider =
        (["brave", "tavily"] as const).find(
          (candidate) =>
            Boolean(configuredCredential(candidate)) &&
            providerIsSupported(candidate, agent, dockerfilePathOverride),
        ) ?? provider;
    }
    if (!provider) return null;
    if (!providerSupported(provider, agent, dockerfilePathOverride)) return null;

    if (isKeylessWebSearchProvider(provider)) {
      // DuckDuckGo brokers no credential and is never implicitly detected —
      // it only reaches here via an explicit NEMOCLAW_WEB_SEARCH_PROVIDER or
      // an existing persisted config, so just accept it.
      deps.note(`  [non-interactive] ${providerSpec(provider).label} requested.`);
      return { fetchEnabled: true, provider };
    }

    const spec = providerSpec(provider);
    const apiKey = configuredCredential(provider);
    if (!apiKey) {
      if (isOptionalCredentialWebSearchProvider(provider)) {
        deps.note(`  [non-interactive] ${spec.label} requested (no key — local instance).`);
        return { fetchEnabled: true, provider };
      }
      if (explicit.specified || existingConfig) {
        throw new Error(
          `${spec.label} requires ${spec.envKey} or a saved ${spec.label} credential in non-interactive mode.`,
        );
      }
      return null;
    }

    deps.note(`  [non-interactive] ${spec.label} requested.`);
    const validation = validateWebSearchApiKey(provider, apiKey);
    if (!validation.ok) {
      console.warn(
        `  ${spec.label} API key validation failed. Web search will be disabled — re-enable it by rerunning ${deps.cliName()} onboard.`,
      );
      if (validation.message) console.warn(`  ${validation.message}`);
      return null;
    }
    stageValidatedCredential(provider, apiKey);
    return { fetchEnabled: true, provider };
  }

  async function configureWebSearch(
    existingConfig: WebSearchConfig | null = null,
    agent: AgentDefinition | null = null,
    dockerfilePathOverride: string | null = null,
  ): Promise<WebSearchConfig | null> {
    if (!agentSupportsWebSearch(agent, dockerfilePathOverride, ROOT)) {
      deps.note(
        `  Web search is not yet supported by ${agent?.displayName ?? "this agent"}. Skipping.`,
      );
      return null;
    }

    existingConfig = normalizeWebSearchConfig(existingConfig);

    if (deps.isNonInteractive()) {
      return configureNonInteractiveWebSearch(existingConfig, agent, dockerfilePathOverride);
    }

    if (existingConfig) return normalizeWebSearchConfig(existingConfig);

    const supportedProviders = WEB_SEARCH_PROVIDERS.filter((provider) =>
      providerIsSupported(provider, agent, dockerfilePathOverride),
    );
    while (true) {
      const provider = await promptWebSearchProvider(supportedProviders);
      if (!provider) return null;

      const apiKey = await ensureValidatedWebSearchCredential(provider);
      if (isBackToSelection(apiKey)) continue;
      if (!apiKey) return null;

      console.log(`  ✓ Enabled ${providerSpec(provider).label}`);
      console.log("");
      return { fetchEnabled: true, provider };
    }
  }

  async function verifyWebSearchInsideSandbox(
    sandboxName: string,
    agent: AgentDefinition | null | undefined,
    provider: WebSearchProvider,
  ): Promise<boolean> {
    return await verifyWebSearchInsideSandboxWithDeps(sandboxName, agent, provider, {
      commandExecutor: deps.commandExecutor,
      cliName: deps.cliName,
      webSearchEnvFor,
      webSearchLabelFor,
    });
  }

  return {
    webSearchProviderForConfig,
    validateWebSearchApiKey,
    validateBraveSearchApiKey,
    validateTavilySearchApiKey,
    promptWebSearchRecovery,
    promptBraveSearchRecovery,
    promptWebSearchApiKey,
    promptBraveSearchApiKey,
    promptWebSearchProvider,
    resolveNonInteractiveWebSearchProvider,
    ensureValidatedWebSearchCredential,
    ensureValidatedBraveSearchCredential,
    configureWebSearch,
    verifyWebSearchInsideSandbox,
  };
}
