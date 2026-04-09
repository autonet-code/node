/**
 * codex-mcp-relay.ts -- Thin MCP stdio server that relays ATN tool calls
 * back to the codex-bridge parent process via TCP.
 *
 * The Codex CLI spawns this as its MCP server (configured via --config).
 * It speaks the MCP protocol (JSON-RPC 2.0 over stdio) to the CLI, and
 * relays tool calls over a TCP connection to the codex-bridge parent.
 *
 * Environment variables:
 *   ATN_RELAY_PORT  — TCP port the parent bridge is listening on
 *   ATN_TOOLS_JSON  — JSON-encoded array of tool definitions
 */

import { createInterface } from "readline"
import { createConnection, type Socket } from "net"

// -- Logging (stderr only, never stdout — that's the MCP protocol) --

function log(msg: string, extra?: Record<string, unknown>): void {
  const parts = ["[mcp-relay]", msg]
  if (extra && Object.keys(extra).length > 0) {
    parts.push(JSON.stringify(extra))
  }
  process.stderr.write(parts.join(" ") + "\n")
}

// -- Tool definitions (passed via env) --

interface ToolDef {
  name: string
  description: string
  input_schema: Record<string, any>
}

const toolsJson = process.env.ATN_TOOLS_JSON || "[]"
let tools: ToolDef[] = []
try {
  tools = JSON.parse(toolsJson)
} catch {
  log("Failed to parse ATN_TOOLS_JSON, no tools available")
}

log("loaded tools", { count: tools.length })

// -- TCP connection to parent bridge --

const relayPort = parseInt(process.env.ATN_RELAY_PORT || "0", 10)
if (!relayPort) {
  log("ATN_RELAY_PORT not set, tool calls will fail")
}

let socket: Socket | null = null
let socketReady = false
const pendingCalls = new Map<string, {
  resolve: (result: any) => void
  reject: (error: Error) => void
}>()

function connectToParent(): Promise<void> {
  return new Promise((resolve, reject) => {
    if (!relayPort) {
      resolve()
      return
    }

    socket = createConnection({ port: relayPort, host: "127.0.0.1" }, () => {
      socketReady = true
      log("connected to parent bridge", { port: relayPort })
      resolve()
    })

    socket.on("error", (err) => {
      log("socket error", { error: err.message })
      socketReady = false
      reject(err)
    })

    socket.on("close", () => {
      log("socket closed")
      socketReady = false
    })

    // Read responses from the parent (NDJSON)
    const socketRl = createInterface({ input: socket })
    socketRl.on("line", (line: string) => {
      try {
        const msg = JSON.parse(line)
        if (msg.type === "tool_result" && msg.call_id) {
          const pending = pendingCalls.get(msg.call_id)
          if (pending) {
            pendingCalls.delete(msg.call_id)
            pending.resolve(msg.result)
          }
        }
      } catch {
        log("bad response from parent", { line: line.slice(0, 200) })
      }
    })
  })
}

async function callTool(name: string, args: Record<string, any>): Promise<any> {
  if (!socket || !socketReady) {
    return { error: "Not connected to parent bridge" }
  }

  const callId = Math.random().toString(36).slice(2, 14)

  return new Promise((resolve, reject) => {
    pendingCalls.set(callId, { resolve, reject })

    const msg = JSON.stringify({
      type: "tool_call",
      call_id: callId,
      name,
      input: args,
    }) + "\n"

    socket!.write(msg, (err) => {
      if (err) {
        pendingCalls.delete(callId)
        reject(err)
      }
    })

    // Timeout after 5 minutes
    setTimeout(() => {
      if (pendingCalls.has(callId)) {
        pendingCalls.delete(callId)
        resolve({ error: `Tool call '${name}' timed out after 5 minutes` })
      }
    }, 300_000)
  })
}

// -- MCP Protocol (JSON-RPC 2.0 over stdio) --

function sendResponse(id: string | number, result: any): void {
  const resp = JSON.stringify({
    jsonrpc: "2.0",
    id,
    result,
  })
  process.stdout.write(resp + "\n")
}

function sendError(id: string | number | null, code: number, message: string): void {
  const resp = JSON.stringify({
    jsonrpc: "2.0",
    id,
    error: { code, message },
  })
  process.stdout.write(resp + "\n")
}

async function handleRequest(req: any): Promise<void> {
  const { id, method, params } = req

  switch (method) {
    case "initialize": {
      sendResponse(id, {
        protocolVersion: "2024-11-05",
        capabilities: {
          tools: {},
        },
        serverInfo: {
          name: "atn-relay",
          version: "1.0.0",
        },
      })
      break
    }

    case "notifications/initialized": {
      // Client notification, no response needed
      log("client initialized")
      break
    }

    case "tools/list": {
      const mcpTools = tools.map((t) => ({
        name: t.name,
        description: t.description,
        inputSchema: t.input_schema,
      }))
      sendResponse(id, { tools: mcpTools })
      break
    }

    case "tools/call": {
      const toolName = params?.name || ""
      const toolArgs = params?.arguments || {}

      log("tool call", { name: toolName })

      try {
        const result = await callTool(toolName, toolArgs)
        const text = typeof result === "string" ? result : JSON.stringify(result, null, 2)
        sendResponse(id, {
          content: [{ type: "text", text }],
        })
      } catch (err) {
        sendResponse(id, {
          content: [{ type: "text", text: `Error: ${err instanceof Error ? err.message : String(err)}` }],
          isError: true,
        })
      }
      break
    }

    case "ping": {
      sendResponse(id, {})
      break
    }

    default: {
      if (id !== undefined && id !== null) {
        sendError(id, -32601, `Method not found: ${method}`)
      }
    }
  }
}

// -- Main --

async function main(): Promise<void> {
  // Connect to parent bridge
  try {
    await connectToParent()
  } catch {
    log("Failed to connect to parent, will run without tool relay")
  }

  // Read MCP requests from stdin
  const rl = createInterface({ input: process.stdin })

  rl.on("line", async (line: string) => {
    const trimmed = line.trim()
    if (!trimmed) return

    try {
      const req = JSON.parse(trimmed)
      await handleRequest(req)
    } catch (err) {
      log("parse error", { error: err instanceof Error ? err.message : String(err) })
      sendError(null, -32700, "Parse error")
    }
  })

  rl.on("close", () => {
    log("stdin closed, shutting down")
    if (socket) socket.destroy()
    process.exit(0)
  })

  log("relay ready", { tools: tools.length, relayPort })
}

main().catch((err) => {
  log("fatal", { error: err.message })
  process.exit(1)
})
