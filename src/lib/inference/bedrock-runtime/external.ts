// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Externally managed Bedrock Runtime adapter support.
 *
 * An external supervisor (for example `../nemoclaw-artha-config/start-bedrock-adapter.sh`
 * next to this repository, or the path in NEMOCLAW_BEDROCK_ADAPTER_START_SCRIPT, run after
 * `aws sso login`) owns the adapter process and its host credentials. It
 * publishes the usual lifecycle files under the NemoClaw state directory and marks the
 * state with `"managedBy": "external"`. NemoClaw then only reuses a healthy external
 * adapter: it never spawns, signals, rewrites, or deletes that adapter or its evidence.
 */

import net from "node:net";
import path from "node:path";

import { BEDROCK_RUNTIME_ADAPTER_PORT } from "../../core/ports";
import { REPOSITORY_ROOT } from "../../core/repository-root";
import { BEDROCK_RUNTIME_ADAPTER_LOOPBACK_HOST } from "../bedrock-runtime";
import {
  DEFAULT_LOCAL_ADAPTER_STATE_DIR,
  type JsonObject,
  localAdapterTokenHash,
  probeLocalAdapterHealth,
  readLocalAdapterJsonFile,
  readLocalAdapterTextFile,
  waitForLocalAdapterHealth,
} from "../local-adapter-lifecycle";

export const BEDROCK_RUNTIME_ADAPTER_EXTERNAL_ENV = "NEMOCLAW_BEDROCK_ADAPTER_EXTERNAL";
export const BEDROCK_RUNTIME_ADAPTER_EXTERNAL_MANAGER = "external";
export const BEDROCK_RUNTIME_ADAPTER_START_SCRIPT_ENV = "NEMOCLAW_BEDROCK_ADAPTER_START_SCRIPT";
export const BEDROCK_RUNTIME_ADAPTER_STATE_FILE = "bedrock-runtime-adapter.json";
export const BEDROCK_RUNTIME_ADAPTER_TOKEN_FILE = "bedrock-runtime-adapter-token";
export const BEDROCK_RUNTIME_ADAPTER_PID_FILE = "bedrock-runtime-adapter.pid";
export const BEDROCK_RUNTIME_ADAPTER_LOG_FILE = "bedrock-runtime-adapter.log";
export const BEDROCK_RUNTIME_ADAPTER_MODEL_CATALOG_HEALTH_PATH = "/health/model-catalog";

/** Files the external supervisor owns; uninstall must preserve them. */
export const EXTERNAL_BEDROCK_RUNTIME_ADAPTER_STATE_ENTRIES: readonly string[] = [
  BEDROCK_RUNTIME_ADAPTER_STATE_FILE,
  BEDROCK_RUNTIME_ADAPTER_TOKEN_FILE,
  BEDROCK_RUNTIME_ADAPTER_PID_FILE,
  BEDROCK_RUNTIME_ADAPTER_LOG_FILE,
];

const ADAPTER_TOKEN = /^[A-Za-z0-9._~+/=-]{32,512}$/u;

export type ExternalBedrockRuntimeAdapterHealthProbe = (
  port: number,
  tokenHash: string,
) => Promise<boolean>;

export type ExternalBedrockRuntimeAdapterResolution =
  | { kind: "not-external" }
  | {
      kind: "healthy";
      adapterPort: number;
      endpointUrl: string | null;
      pid: number | null;
      profile: string | null;
      region: string | null;
      token: string;
      tokenHash: string;
    }
  | { kind: "unhealthy"; adapterPort: number; reason: string };

export function externalBedrockRuntimeAdapterStartCommand(
  root: string = REPOSITORY_ROOT,
  env: NodeJS.ProcessEnv = process.env,
): string {
  const configured = String(env[BEDROCK_RUNTIME_ADAPTER_START_SCRIPT_ENV] ?? "").trim();
  if (configured) return configured;
  // The supervisor scripts live in a private checkout next to this repository.
  return path.join(path.dirname(root), "nemoclaw-artha-config", "start-bedrock-adapter.sh");
}

export function isExternalBedrockRuntimeAdapterForced(
  env: NodeJS.ProcessEnv = process.env,
): boolean {
  const value = String(env[BEDROCK_RUNTIME_ADAPTER_EXTERNAL_ENV] ?? "")
    .trim()
    .toLowerCase();
  return value === "1" || value === "true" || value === "yes";
}

export function isExternalBedrockRuntimeAdapterState(value: unknown): boolean {
  return (
    Boolean(value) &&
    typeof value === "object" &&
    !Array.isArray(value) &&
    (value as JsonObject).managedBy === BEDROCK_RUNTIME_ADAPTER_EXTERNAL_MANAGER
  );
}

/** True when the state file or the environment hands adapter ownership to an external supervisor. */
export function isExternalBedrockRuntimeAdapterMode(
  state: unknown,
  env: NodeJS.ProcessEnv = process.env,
): boolean {
  return isExternalBedrockRuntimeAdapterState(state) || isExternalBedrockRuntimeAdapterForced(env);
}

export function bedrockRuntimeAdapterStatePaths(
  stateDir: string = DEFAULT_LOCAL_ADAPTER_STATE_DIR,
): {
  statePath: string;
  tokenPath: string;
  pidPath: string;
} {
  return {
    statePath: path.join(stateDir, BEDROCK_RUNTIME_ADAPTER_STATE_FILE),
    tokenPath: path.join(stateDir, BEDROCK_RUNTIME_ADAPTER_TOKEN_FILE),
    pidPath: path.join(stateDir, BEDROCK_RUNTIME_ADAPTER_PID_FILE),
  };
}

/** Whether `stateDir` currently holds externally managed adapter evidence (or the env forces it). */
export function readExternalBedrockRuntimeAdapterMode(
  stateDir: string = DEFAULT_LOCAL_ADAPTER_STATE_DIR,
  env: NodeJS.ProcessEnv = process.env,
): boolean {
  return isExternalBedrockRuntimeAdapterMode(
    readLocalAdapterJsonFile(bedrockRuntimeAdapterStatePaths(stateDir).statePath),
    env,
  );
}

function isValidPort(value: unknown): value is number {
  return Number.isSafeInteger(value) && (value as number) >= 1 && (value as number) <= 65_535;
}

function optionalString(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

/** Reusable adapter tokens are opaque bearer strings without whitespace. */
export function isReusableBedrockRuntimeAdapterToken(token: string | null | undefined): boolean {
  return typeof token === "string" && ADAPTER_TOKEN.test(token);
}

export function defaultExternalBedrockRuntimeAdapterHealthProbe(
  port: number,
  tokenHash: string,
): Promise<boolean> {
  return waitForLocalAdapterHealth(
    () =>
      probeLocalAdapterHealth({
        host: BEDROCK_RUNTIME_ADAPTER_LOOPBACK_HOST,
        port,
        path: BEDROCK_RUNTIME_ADAPTER_MODEL_CATALOG_HEALTH_PATH,
        expectedTokenHash: tokenHash,
      }),
    { attempts: 3, intervalMs: 100 },
  );
}

/**
 * Decide whether an already-read adapter state/token pair describes a healthy external adapter.
 * The check is read-only: it never writes, removes, or signals anything.
 */
export async function resolveExternalBedrockRuntimeAdapter(options: {
  state: unknown;
  token: string | null;
  env?: NodeJS.ProcessEnv;
  expectedPort?: number;
  probeHealth?: ExternalBedrockRuntimeAdapterHealthProbe;
}): Promise<ExternalBedrockRuntimeAdapterResolution> {
  const env = options.env ?? process.env;
  if (!isExternalBedrockRuntimeAdapterMode(options.state, env)) return { kind: "not-external" };
  const expectedPort = options.expectedPort ?? BEDROCK_RUNTIME_ADAPTER_PORT;
  const state =
    options.state && typeof options.state === "object" && !Array.isArray(options.state)
      ? (options.state as JsonObject)
      : {};
  const adapterPort = isValidPort(state.adapterPort) ? state.adapterPort : expectedPort;
  if (adapterPort !== expectedPort) {
    return {
      kind: "unhealthy",
      adapterPort,
      reason: `its state records port ${String(adapterPort)} but NemoClaw routes sandboxes to port ${String(expectedPort)}; restart it on ${String(expectedPort)} or export NEMOCLAW_BEDROCK_RUNTIME_ADAPTER_PORT=${String(adapterPort)}`,
    };
  }
  const token = options.token?.trim() || null;
  if (!token) {
    return { kind: "unhealthy", adapterPort, reason: "its token file is missing or empty" };
  }
  const tokenHash = localAdapterTokenHash(token);
  if (typeof state.tokenHash === "string" && state.tokenHash !== tokenHash) {
    return {
      kind: "unhealthy",
      adapterPort,
      reason: "its token file does not match the tokenHash recorded in its state file",
    };
  }
  const probe = options.probeHealth ?? defaultExternalBedrockRuntimeAdapterHealthProbe;
  let healthy = false;
  try {
    healthy = await probe(adapterPort, tokenHash);
  } catch {
    healthy = false;
  }
  if (!healthy) {
    return {
      kind: "unhealthy",
      adapterPort,
      reason: "no adapter with the recorded token answered its health check",
    };
  }
  return {
    kind: "healthy",
    adapterPort,
    endpointUrl: optionalString(state.endpointUrl),
    pid:
      Number.isSafeInteger(state.pid) && (state.pid as number) > 0 ? (state.pid as number) : null,
    profile: optionalString(state.profile),
    region: optionalString(state.region),
    token,
    tokenHash,
  };
}

/** Read the external adapter evidence from `stateDir` and resolve its health. */
export async function probeExternalBedrockRuntimeAdapter(
  options: {
    stateDir?: string;
    env?: NodeJS.ProcessEnv;
    expectedPort?: number;
    probeHealth?: ExternalBedrockRuntimeAdapterHealthProbe;
  } = {},
): Promise<ExternalBedrockRuntimeAdapterResolution> {
  const paths = bedrockRuntimeAdapterStatePaths(options.stateDir);
  const state = readLocalAdapterJsonFile(paths.statePath);
  const env = options.env ?? process.env;
  if (!isExternalBedrockRuntimeAdapterMode(state, env)) return { kind: "not-external" };
  return resolveExternalBedrockRuntimeAdapter({
    state,
    token: readLocalAdapterTextFile(paths.tokenPath),
    env,
    expectedPort: options.expectedPort,
    probeHealth: options.probeHealth,
  });
}

export function formatExternalBedrockRuntimeAdapterUnavailable(
  result: Extract<ExternalBedrockRuntimeAdapterResolution, { kind: "unhealthy" }>,
  startCommand: string = externalBedrockRuntimeAdapterStartCommand(),
): string {
  return `External Bedrock adapter is not running/healthy on port ${String(result.adapterPort)} (${result.reason}). Start it with: ${startCommand}`;
}

export class ExternalBedrockRuntimeAdapterUnavailableError extends Error {
  readonly adapterPort: number;

  constructor(result: Extract<ExternalBedrockRuntimeAdapterResolution, { kind: "unhealthy" }>) {
    super(formatExternalBedrockRuntimeAdapterUnavailable(result));
    this.name = "ExternalBedrockRuntimeAdapterUnavailableError";
    this.adapterPort = result.adapterPort;
  }
}

export type LoopbackPortState = "listening" | "free" | "unknown";

/**
 * Classify a loopback TCP port without sending data. Only a refused connection proves the
 * port is free; timeouts and other errors stay "unknown" so callers refuse destructive work.
 */
export function probeLoopbackPortState(
  port: number,
  options: { host?: string; timeoutMs?: number } = {},
): Promise<LoopbackPortState> {
  return new Promise((resolve) => {
    let settled = false;
    const socket = net.createConnection({
      host: options.host ?? BEDROCK_RUNTIME_ADAPTER_LOOPBACK_HOST,
      port,
    });
    const settle = (state: LoopbackPortState) => {
      if (settled) return;
      settled = true;
      socket.destroy();
      resolve(state);
    };
    socket.setTimeout(options.timeoutMs ?? 1_000);
    socket.once("connect", () => settle("listening"));
    socket.once("timeout", () => settle("unknown"));
    socket.once("error", (error: NodeJS.ErrnoException) =>
      settle(error.code === "ECONNREFUSED" ? "free" : "unknown"),
    );
  });
}

export type BedrockRuntimeAdapterHostReport = {
  status: "ok" | "fail";
  managedBy: "external" | "nemoclaw" | "none";
  detail: string;
  hint?: string;
};

/**
 * Read-only host-side health summary for status/doctor/start diagnostics. It never starts
 * an adapter: an external adapter needs an interactive `aws sso login` first.
 */
export async function inspectBedrockRuntimeAdapterHost(
  options: {
    stateDir?: string;
    env?: NodeJS.ProcessEnv;
    expectedPort?: number;
    probeHealth?: ExternalBedrockRuntimeAdapterHealthProbe;
    startCommand?: string;
  } = {},
): Promise<BedrockRuntimeAdapterHostReport> {
  const expectedPort = options.expectedPort ?? BEDROCK_RUNTIME_ADAPTER_PORT;
  const startCommand = options.startCommand ?? externalBedrockRuntimeAdapterStartCommand();
  const probeHealth = options.probeHealth ?? defaultExternalBedrockRuntimeAdapterHealthProbe;
  const external = await probeExternalBedrockRuntimeAdapter({
    ...options,
    expectedPort,
    probeHealth,
  });
  if (external.kind === "healthy") {
    const facts = [
      external.profile ? `profile ${external.profile}` : null,
      external.region ? `region ${external.region}` : null,
      external.pid ? `pid ${String(external.pid)}` : null,
    ].filter(Boolean);
    return {
      status: "ok",
      managedBy: "external",
      detail: `external adapter healthy on 127.0.0.1:${String(external.adapterPort)}${facts.length > 0 ? ` (${facts.join(", ")})` : ""}`,
    };
  }
  if (external.kind === "unhealthy") {
    return {
      status: "fail",
      managedBy: "external",
      detail: `external adapter not running/healthy on port ${String(external.adapterPort)}: ${external.reason}`,
      hint: `run \`aws sso login\` if needed, then start it with ${startCommand}`,
    };
  }
  const paths = bedrockRuntimeAdapterStatePaths(options.stateDir);
  const state = readLocalAdapterJsonFile(paths.statePath);
  const token = readLocalAdapterTextFile(paths.tokenPath);
  if (!state && !token) {
    return {
      status: "fail",
      managedBy: "none",
      detail: `no Bedrock Runtime adapter state found for port ${String(expectedPort)}`,
      hint: `start the external adapter with ${startCommand}, or re-run onboarding`,
    };
  }
  const port = isValidPort(state?.adapterPort) ? state.adapterPort : expectedPort;
  let healthy = false;
  if (token) {
    try {
      healthy = await probeHealth(port, localAdapterTokenHash(token));
    } catch {
      healthy = false;
    }
  }
  return healthy
    ? {
        status: "ok",
        managedBy: "nemoclaw",
        detail: `NemoClaw-managed adapter healthy on 127.0.0.1:${String(port)}`,
      }
    : {
        status: "fail",
        managedBy: "nemoclaw",
        detail: `NemoClaw-managed adapter is not running/healthy on port ${String(port)}`,
        hint: `start the external adapter with ${startCommand} (it reuses the existing token), or re-run onboarding`,
      };
}
