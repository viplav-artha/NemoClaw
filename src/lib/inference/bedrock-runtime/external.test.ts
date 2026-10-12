// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import crypto from "node:crypto";
import fs from "node:fs";
import net from "node:net";
import os from "node:os";
import path from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import {
  externalBedrockRuntimeAdapterStartCommand,
  formatExternalBedrockRuntimeAdapterUnavailable,
  inspectBedrockRuntimeAdapterHost,
  isExternalBedrockRuntimeAdapterMode,
  isReusableBedrockRuntimeAdapterToken,
  probeExternalBedrockRuntimeAdapter,
  probeLoopbackPortState,
  resolveExternalBedrockRuntimeAdapter,
} from "./external";

const TOKEN = "a1".repeat(24);
const TOKEN_HASH = crypto.createHash("sha256").update(TOKEN).digest("hex");
const homes: string[] = [];

function stateDirWith(files: { state?: unknown; token?: string }): string {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), "nemoclaw-bedrock-external-"));
  homes.push(home);
  if (files.state !== undefined) {
    fs.writeFileSync(path.join(home, "bedrock-runtime-adapter.json"), JSON.stringify(files.state), {
      mode: 0o600,
    });
  }
  if (files.token !== undefined) {
    fs.writeFileSync(path.join(home, "bedrock-runtime-adapter-token"), `${files.token}\n`, {
      mode: 0o600,
    });
  }
  return home;
}

afterEach(() => {
  for (const home of homes.splice(0)) fs.rmSync(home, { recursive: true, force: true });
});

describe("external Bedrock Runtime adapter", () => {
  it("detects external mode from the state marker or the environment override", () => {
    expect(isExternalBedrockRuntimeAdapterMode({ managedBy: "external" }, {})).toBe(true);
    expect(isExternalBedrockRuntimeAdapterMode({ managedBy: "nemoclaw" }, {})).toBe(false);
    expect(
      isExternalBedrockRuntimeAdapterMode(null, { NEMOCLAW_BEDROCK_ADAPTER_EXTERNAL: "1" }),
    ).toBe(true);
    expect(isExternalBedrockRuntimeAdapterMode(null, {})).toBe(false);
  });

  it("reuses a healthy external adapter by its token hash without other identity checks", async () => {
    const probeHealth = vi.fn(async () => true);
    await expect(
      resolveExternalBedrockRuntimeAdapter({
        state: {
          managedBy: "external",
          adapterPort: 11_436,
          tokenHash: TOKEN_HASH,
          credentialHash: "c".repeat(64),
          region: "us-east-1",
          profile: "dev",
          pid: 77,
        },
        token: TOKEN,
        env: {},
        expectedPort: 11_436,
        probeHealth,
      }),
    ).resolves.toEqual({
      kind: "healthy",
      adapterPort: 11_436,
      endpointUrl: null,
      pid: 77,
      profile: "dev",
      region: "us-east-1",
      token: TOKEN,
      tokenHash: TOKEN_HASH,
    });
    expect(probeHealth).toHaveBeenCalledWith(11_436, TOKEN_HASH);
  });

  it.each([
    [{ managedBy: "external" }, null, "token file is missing"],
    [{ managedBy: "external", tokenHash: "0".repeat(64) }, TOKEN, "does not match the tokenHash"],
    [{ managedBy: "external", adapterPort: 11_999 }, TOKEN, "records port 11999"],
  ])("reports an unusable external adapter (%j)", async (state, token, reason) => {
    const probeHealth = vi.fn(async () => true);
    const result = await resolveExternalBedrockRuntimeAdapter({
      state,
      token,
      env: {},
      expectedPort: 11_436,
      probeHealth,
    });
    expect(result).toMatchObject({ kind: "unhealthy" });
    expect(result.kind === "unhealthy" ? result.reason : "").toContain(reason);
    expect(probeHealth).not.toHaveBeenCalled();
  });

  it("formats an actionable error naming the start script", async () => {
    const result = await resolveExternalBedrockRuntimeAdapter({
      state: { managedBy: "external" },
      token: TOKEN,
      env: {},
      expectedPort: 11_436,
      probeHealth: async () => false,
    });
    if (result.kind !== "unhealthy") throw new Error("expected unhealthy");
    expect(formatExternalBedrockRuntimeAdapterUnavailable(result, "/repo/start.sh")).toBe(
      "External Bedrock adapter is not running/healthy on port 11436 (no adapter with the recorded token answered its health check). Start it with: /repo/start.sh",
    );
    expect(externalBedrockRuntimeAdapterStartCommand("/work/repo", {})).toBe(
      "/work/nemoclaw-artha-config/start-bedrock-adapter.sh",
    );
    expect(
      externalBedrockRuntimeAdapterStartCommand("/work/repo", {
        NEMOCLAW_BEDROCK_ADAPTER_START_SCRIPT: "/opt/artha/start.sh",
      }),
    ).toBe("/opt/artha/start.sh");
  });

  it("reads external evidence from a state directory", async () => {
    const stateDir = stateDirWith({
      state: { managedBy: "external", adapterPort: 11_436, tokenHash: TOKEN_HASH },
      token: TOKEN,
    });
    await expect(
      probeExternalBedrockRuntimeAdapter({
        stateDir,
        env: {},
        expectedPort: 11_436,
        probeHealth: async () => true,
      }),
    ).resolves.toMatchObject({ kind: "healthy", token: TOKEN });
    await expect(
      probeExternalBedrockRuntimeAdapter({
        stateDir: stateDirWith({ state: { version: 2 }, token: TOKEN }),
        env: {},
      }),
    ).resolves.toEqual({ kind: "not-external" });
  });

  it("summarizes host adapter state for diagnostics without starting anything", async () => {
    const external = await inspectBedrockRuntimeAdapterHost({
      stateDir: stateDirWith({
        state: { managedBy: "external", tokenHash: TOKEN_HASH },
        token: TOKEN,
      }),
      env: {},
      expectedPort: 11_436,
      probeHealth: async () => false,
      startCommand: "/repo/start.sh",
    });
    expect(external).toMatchObject({ status: "fail", managedBy: "external" });
    expect(external.hint).toContain("/repo/start.sh");

    const missing = await inspectBedrockRuntimeAdapterHost({
      stateDir: stateDirWith({}),
      env: {},
      expectedPort: 11_436,
      probeHealth: async () => true,
    });
    expect(missing).toMatchObject({ status: "fail", managedBy: "none" });

    const managed = await inspectBedrockRuntimeAdapterHost({
      stateDir: stateDirWith({ state: { version: 2, adapterPort: 11_436 }, token: TOKEN }),
      env: {},
      expectedPort: 11_436,
      probeHealth: async () => true,
    });
    expect(managed).toMatchObject({ status: "ok", managedBy: "nemoclaw" });
  });

  it("accepts only opaque bearer-like tokens for reuse", () => {
    expect(isReusableBedrockRuntimeAdapterToken(TOKEN)).toBe(true);
    expect(isReusableBedrockRuntimeAdapterToken("short")).toBe(false);
    expect(isReusableBedrockRuntimeAdapterToken(`${TOKEN} extra`)).toBe(false);
    expect(isReusableBedrockRuntimeAdapterToken(null)).toBe(false);
  });

  it("classifies a loopback port as listening or free", async () => {
    const server = net.createServer();
    await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
    const address = server.address();
    if (!address || typeof address === "string") throw new Error("expected TCP address");
    try {
      await expect(probeLoopbackPortState(address.port)).resolves.toBe("listening");
    } finally {
      await new Promise<void>((resolve) => server.close(() => resolve()));
    }
    await expect(probeLoopbackPortState(address.port)).resolves.toBe("free");
  });
});
