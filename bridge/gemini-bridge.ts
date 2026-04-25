/**
 * gemini-bridge.ts -- stdin/stdout bridge between ATN and Gemini CLI via ACP.
 *
 * Architecture:
 *   ATN (Python)  <-- NDJSON over stdin/stdout -->  this bridge
 *                                                       |
 *                                                       v
 *                              Gemini CLI subprocess  <-- ACP (JSON-RPC 2.0) over stdio
 *                              (gemini --experimental-acp)
 *
 * The Python side speaks the same NDJSON protocol it uses for the Claude and
 * Codex bridges (request types: create / orchestrate / shutdown).  We translate
 * those into ACP calls (initialize / authenticate / newSession / loadSession /
 * prompt / cancel) on the Gemini side.
 *
 * Auth: the user runs `gemini` interactively once and signs in via OAuth.
 * The credential is stored under ~/.gemini/.  ACP `authenticate` reuses that
 * stored token, so the bridge picks up subscription auth automatically.
 *
 * Status: SKELETON.  Step 1 scaffolds the subprocess, ACP handshake, and an
 * idle loop that responds to Python requests with a "not implemented yet"
 * stub.  The prompt path, streaming, MCP tool relay, and cancel are wired in
 * later steps once the handshake is verified.
 */

import { spawn, type ChildProcess } from "child_process"
import { createInterface } from "readline"
import {
  ClientSideConnection,
  PROTOCOL_VERSION,
  ndJsonStream,
  type Client,
  type RequestPermissionRequest,
  type RequestPermissionResponse,
  type ReadTextFileRequest,
  type ReadTextFileResponse,
  type WriteTextFileRequest,
  type WriteTextFileResponse,
  type SessionNotification,
} from "@zed-industries/agent-client-protocol"
import { Readable, Writable } from "stream"

// -- Logging (stderr only) --

function log(msg: string, extra?: Record<string, unknown>): void {
  const parts = ["[gemini-bridge]", msg]
  if (extra && Object.keys(extra).length > 0) {
    parts.push(JSON.stringify(extra))
  }
  process.stderr.write(parts.join(" ") + "\n")
}

const EVENT_PREFIX = "@@EVENT@@"

function emitEvent(event: { type: string; [key: string]: unknown }): void {
  process.stderr.write(`${EVENT_PREFIX}${JSON.stringify(event)}\n`)
}

// -- Global error handlers --

process.on("unhandledRejection", (reason: unknown) => {
  log("unhandled rejection (kept alive)", {
    error: reason instanceof Error ? reason.message : String(reason),
  })
})

process.on("uncaughtException", (err: Error) => {
  log("uncaught exception (kept alive)", { error: err.message })
})

// -- Gemini subprocess + ACP connection --

interface GeminiSession {
  sessionId: string
  createdAt: number
}

class GeminiAgent {
  private proc: ChildProcess | null = null
  private connection: ClientSideConnection | null = null
  private sessions = new Map<string, GeminiSession>()
  private authenticated = false
  private authMethods: Array<{ id: string; name: string; description?: string | null }> = []
  private agentCapabilities: Record<string, unknown> = {}

  /** Spawn the Gemini CLI in ACP mode and complete the initialize handshake. */
  async start(): Promise<void> {
    if (this.proc) return

    log("spawning gemini --experimental-acp")

    // On Windows, "gemini" is a .cmd shim — needs shell:true to invoke via cmd.
    const isWindows = process.platform === "win32"
    this.proc = spawn("gemini", ["--experimental-acp"], {
      stdio: ["pipe", "pipe", "pipe"],
      shell: isWindows,
    })

    this.proc.on("exit", (code, signal) => {
      log("gemini subprocess exited", { code, signal })
      this.proc = null
      this.connection = null
    })

    // Drain stderr for visibility — the CLI writes log lines there.
    // (Side effect: attaching this listener also puts the corresponding stdout
    // into flowing mode, which on Node 24 is required for Readable.toWeb()
    // to start consuming — without it, the ACP handshake hangs.)
    if (this.proc.stderr) {
      const rl = createInterface({ input: this.proc.stderr })
      rl.on("line", (line) => log("gemini stderr", { line: line.slice(0, 300) }))
    }
    // Touching stdout's "data" event is what kicks Readable.toWeb() into
    // motion; we don't actually consume here, just trigger flowing mode.
    if (this.proc.stdout) {
      this.proc.stdout.on("error", (err) => log("gemini stdout error", { error: err.message }))
    }

    // Wrap the subprocess stdio as web streams the way ACP expects.
    const stdin = this.proc.stdin
    const stdout = this.proc.stdout
    if (!stdin || !stdout) {
      throw new Error("gemini subprocess missing stdin/stdout")
    }

    // Gemini's ACP server splits incoming lines on os.EOL — which on Windows
    // is \r\n.  ACP's ndJsonStream writes \n only, so on Windows we wrap the
    // writable in a TransformStream that translates \n → \r\n before bytes
    // hit the subprocess.
    const baseWritable = Writable.toWeb(stdin) as WritableStream<Uint8Array>
    const readable = Readable.toWeb(stdout) as ReadableStream<Uint8Array>

    let writable = baseWritable
    if (isWindows) {
      const transform = new TransformStream<Uint8Array, Uint8Array>({
        transform(chunk, controller) {
          const out: number[] = []
          for (let i = 0; i < chunk.length; i++) {
            const b = chunk[i]
            if (b === 0x0a /* \n */ && (i === 0 || chunk[i - 1] !== 0x0d)) {
              out.push(0x0d, 0x0a)
            } else {
              out.push(b)
            }
          }
          controller.enqueue(new Uint8Array(out))
        },
      })
      // Pipe the transform's readable into the actual subprocess stdin.
      transform.readable.pipeTo(baseWritable).catch((err) => {
        log("crlf transform pipe error", { error: err?.message ?? String(err) })
      })
      writable = transform.writable
    }

    const stream = ndJsonStream(writable, readable)

    // Build the Client handler.  Most methods are stubs at this stage —
    // we'll wire fs reads/writes and permission prompts in later steps.
    const clientFactory = (_conn: ClientSideConnection): Client => ({
      sessionUpdate: async (params: SessionNotification): Promise<void> => {
        // Forward streaming updates to Python via @@EVENT@@ on stderr.
        // The shape varies (text deltas, tool calls, thinking, etc) — we just
        // echo for now and let Python decide what to surface.
        emitEvent({ type: "session_update", ...(params as unknown as Record<string, unknown>) })
      },
      requestPermission: async (
        params: RequestPermissionRequest,
      ): Promise<RequestPermissionResponse> => {
        // Auto-deny for now.  The orchestrator can later approve specific
        // tool patterns; ATN's existing approval model is per-tool, not per-call.
        log("requestPermission auto-denied", { tool: (params as any).toolCall?.title })
        return { outcome: { outcome: "cancelled" } }
      },
      readTextFile: async (params: ReadTextFileRequest): Promise<ReadTextFileResponse> => {
        // Filesystem capability — gemini will only call this if we advertise
        // fs.readTextFile=true.  We don't yet, so this should never fire.
        log("readTextFile unexpectedly called", params as unknown as Record<string, unknown>)
        return { content: "" }
      },
      writeTextFile: async (
        _params: WriteTextFileRequest,
      ): Promise<WriteTextFileResponse> => {
        log("writeTextFile unexpectedly called")
        return {}
      },
    })

    this.connection = new ClientSideConnection(clientFactory, stream)

    // Initialize handshake.  Gemini's bundled ACP schema requires the `fs`
    // capability to be present (even if both flags are false), unlike newer
    // ACP versions where it's optional.  Pass it explicitly to satisfy
    // Gemini's validator.
    const initRes = await this.connection.initialize({
      protocolVersion: PROTOCOL_VERSION,
      clientCapabilities: {
        fs: { readTextFile: false, writeTextFile: false },
      },
    })

    this.authMethods = (initRes.authMethods || []).map((m) => ({
      id: m.id,
      name: m.name,
      description: (m as any).description ?? null,
    }))
    this.agentCapabilities = (initRes.agentCapabilities || {}) as Record<string, unknown>

    log("initialize ok", {
      protocolVersion: initRes.protocolVersion,
      authMethods: this.authMethods.map((m) => m.id),
      loadSession: (this.agentCapabilities as any).loadSession ?? false,
    })

    emitEvent({
      type: "initialized",
      protocol_version: initRes.protocolVersion,
      auth_methods: this.authMethods,
      agent_capabilities: this.agentCapabilities,
    })
  }

  /** Run authenticate against the agent's stored OAuth credential. */
  async authenticate(methodId?: string): Promise<{ ok: boolean; method?: string; error?: string }> {
    if (!this.connection) return { ok: false, error: "not started" }
    if (this.authenticated) return { ok: true, method: methodId }

    // Prefer caller-specified, else first-advertised, else "oauth-personal".
    const id =
      methodId ||
      this.authMethods.find((m) => /oauth/i.test(m.id))?.id ||
      this.authMethods[0]?.id ||
      "oauth-personal"

    try {
      await this.connection.authenticate({ methodId: id })
      this.authenticated = true
      log("authenticated", { method: id })
      return { ok: true, method: id }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err)
      log("authenticate failed", { method: id, error: msg })
      return { ok: false, method: id, error: msg }
    }
  }

  /** Open a new session.  Falls back to authenticate if creds are stale. */
  async newSession(cwd?: string): Promise<{ ok: boolean; sessionId?: string; error?: string }> {
    if (!this.connection) return { ok: false, error: "not started" }

    const params = {
      cwd: cwd || process.cwd(),
      mcpServers: [],
    }

    try {
      const res = await this.connection.newSession(params)
      const sid = (res as any).sessionId as string
      this.sessions.set(sid, { sessionId: sid, createdAt: Date.now() })
      log("newSession ok", { sessionId: sid })
      return { ok: true, sessionId: sid }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err)
      log("newSession failed", { error: msg })
      return { ok: false, error: msg }
    }
  }

  async stop(): Promise<void> {
    if (this.proc) {
      try {
        this.proc.kill()
      } catch {
        /* ignore */
      }
      this.proc = null
    }
    this.connection = null
    this.sessions.clear()
    this.authenticated = false
  }

  isAlive(): boolean {
    return this.proc !== null && this.connection !== null
  }
}

// -- Bridge protocol (Python <-> this process) --

type BridgeRequest =
  | { id: string; type: "ping" }
  | { id: string; type: "initialize" }
  | { id: string; type: "authenticate"; method?: string }
  | { id: string; type: "new_session"; cwd?: string }
  | { id: string; type: "shutdown" }

type BridgeResponse = {
  id: string
  ok: boolean
  error?: string
  [key: string]: unknown
}

function respond(resp: BridgeResponse): void {
  process.stdout.write(JSON.stringify(resp) + "\n")
}

const agent = new GeminiAgent()

async function handleRequest(req: BridgeRequest): Promise<void> {
  try {
    switch (req.type) {
      case "ping":
        respond({ id: req.id, ok: true, alive: agent.isAlive() })
        return

      case "initialize":
        await agent.start()
        respond({ id: req.id, ok: true })
        return

      case "authenticate": {
        await agent.start()
        const res = await agent.authenticate(req.method)
        respond({ id: req.id, ok: res.ok, method: res.method, error: res.error })
        return
      }

      case "new_session": {
        await agent.start()
        const res = await agent.newSession(req.cwd)
        respond({ id: req.id, ok: res.ok, session_id: res.sessionId, error: res.error })
        return
      }

      case "shutdown":
        await agent.stop()
        respond({ id: req.id, ok: true })
        process.exit(0)
        return

      default: {
        const t = (req as { type?: string }).type
        respond({ id: (req as { id: string }).id, ok: false, error: `unknown request type: ${t}` })
      }
    }
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err)
    respond({ id: req.id, ok: false, error: msg })
  }
}

// -- Read NDJSON requests from stdin --

const rl = createInterface({ input: process.stdin })
rl.on("line", (line) => {
  const trimmed = line.trim()
  if (!trimmed) return
  let req: BridgeRequest
  try {
    req = JSON.parse(trimmed) as BridgeRequest
  } catch (err) {
    log("bad request JSON", { line: trimmed.slice(0, 200) })
    return
  }
  void handleRequest(req)
})

rl.on("close", () => {
  log("stdin closed; shutting down")
  void agent.stop().then(() => process.exit(0))
})

log("ready", { node: process.version })
