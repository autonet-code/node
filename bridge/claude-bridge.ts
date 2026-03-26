/**
 * claude-bridge.ts -- stdin/stdout bridge between ATN and Claude Agent SDK.
 *
 * Protocol (NDJSON over stdin/stdout):
 *   stdin  (ATN -> bridge): JSON requests + tool_result messages
 *   stdout (bridge -> ATN): JSON responses + tool_call messages
 *   stderr: debug logging + @@EVENT@@ streaming events
 *
 * Request types:
 *   create      — single-turn LLM call (maxTurns=1, no tools)
 *   orchestrate — multi-turn with ATN tools as in-process MCP server
 *   delete      — clean up a session
 *   shutdown    — graceful exit
 */

import { query, tool, createSdkMcpServer, getSessionMessages } from "@anthropic-ai/claude-agent-sdk"
import type { Query, SDKMessage, SDKUserMessage, Options } from "@anthropic-ai/claude-agent-sdk"
import { z } from "zod"
import { randomUUID } from "crypto"
import { execSync } from "child_process"
import { existsSync } from "fs"
import { fileURLToPath } from "url"
import { join, dirname } from "path"
import { createInterface } from "readline"

// -- Strip nesting guard --
// The SDK spawns a Claude Code subprocess that refuses to start if
// CLAUDECODE is set.  Clear it so the bridge can operate normally.
delete process.env.CLAUDECODE

// -- Logging (stderr only) --

function log(msg: string, extra?: Record<string, unknown>): void {
  const parts = ["[bridge]", msg]
  if (extra && Object.keys(extra).length > 0) {
    parts.push(JSON.stringify(extra))
  }
  process.stderr.write(parts.join(" ") + "\n")
}

// -- Streaming events (@@EVENT@@ on stderr) --

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
  log("uncaught exception (kept alive)", {
    error: err.message,
  })
})

// -- Claude executable resolution --

function resolveClaudeExecutable(): string {
  // 1. SDK's bundled cli.js
  try {
    const sdkPath = fileURLToPath(import.meta.resolve("@anthropic-ai/claude-agent-sdk"))
    const sdkCliJs = join(dirname(sdkPath), "cli.js")
    if (existsSync(sdkCliJs)) return sdkCliJs
  } catch {}

  // 2. System-installed claude binary
  try {
    const cmd = process.platform === "win32" ? "where claude" : "which claude"
    const claudePath = execSync(cmd, { encoding: "utf-8" }).trim()
    if (claudePath && existsSync(claudePath.split("\n")[0])) return claudePath.split("\n")[0]
  } catch {}

  throw new Error("Could not find Claude Code executable. Install via: npm install -g @anthropic-ai/claude-code")
}

const claudeExecutable = resolveClaudeExecutable()
log("resolved claude executable", { path: claudeExecutable })

function mapModelToClaudeModel(model: string): string {
  // Accept full model IDs (e.g. "claude-sonnet-4-20250514") or family
  // names (e.g. "sonnet").  Full IDs are passed through as-is.
  if (model.startsWith("claude-")) return model
  if (model.includes("opus")) return "opus"
  if (model.includes("haiku")) return "haiku"
  return "sonnet"
}

// -- AsyncQueue --

class AsyncQueue<T> implements AsyncIterable<T> {
  private queue: T[] = []
  private resolve: ((value: IteratorResult<T>) => void) | null = null
  private finished = false

  push(item: T): void {
    if (this.finished) throw new Error("Queue is closed")
    if (this.resolve) {
      const r = this.resolve
      this.resolve = null
      r({ value: item, done: false })
    } else {
      this.queue.push(item)
    }
  }

  done(): void {
    this.finished = true
    if (this.resolve) {
      const r = this.resolve
      this.resolve = null
      r({ value: undefined as any, done: true })
    }
  }

  [Symbol.asyncIterator](): AsyncIterator<T> {
    return {
      next: (): Promise<IteratorResult<T>> => {
        if (this.queue.length > 0) {
          return Promise.resolve({ value: this.queue.shift()!, done: false })
        }
        if (this.finished) {
          return Promise.resolve({ value: undefined as any, done: true })
        }
        return new Promise((resolve) => {
          this.resolve = resolve
        })
      },
    }
  }
}

// -- Tool relay --
// During an orchestrate request, ATN tool calls are relayed to Python for
// execution.  The bridge sends tool_call on stdout, Python responds with
// tool_result on stdin.  Pending calls are tracked by call_id.

const pendingToolCalls = new Map<string, {
  resolve: (result: any) => void
  reject: (error: Error) => void
}>()

function relayToolCall(name: string, input: unknown): Promise<any> {
  const callId = randomUUID().slice(0, 12)
  emitEvent({ type: "tool_use_start", tool_use_id: callId, tool_name: name, input })
  return new Promise((resolve, reject) => {
    pendingToolCalls.set(callId, { resolve, reject })
    const msg = JSON.stringify({
      type: "tool_call",
      call_id: callId,
      name,
      input,
    })
    process.stdout.write(msg + "\n")
    log("tool_call relayed", { callId, name })
  })
}

function handleToolResult(callId: string, result: any): void {
  const pending = pendingToolCalls.get(callId)
  if (pending) {
    pendingToolCalls.delete(callId)
    pending.resolve(result)
    const isError = result != null && typeof result === "object" && !!result.error
    emitEvent({ type: "tool_use_result", tool_use_id: callId, is_error: isError })
    log("tool_result received", { callId })
  } else {
    log("tool_result for unknown call_id", { callId })
  }
}

// -- JSON Schema to Zod conversion --
// The SDK tool() requires Zod schemas.  ATN tools use JSON Schema.
// Convert at runtime.

function jsonSchemaToZod(schema: Record<string, any>): z.ZodType {
  if (!schema || !schema.properties) {
    return z.object({})
  }

  const shape: Record<string, z.ZodType> = {}
  const required = new Set(schema.required || [])

  for (const [key, prop] of Object.entries<any>(schema.properties)) {
    let field: z.ZodType

    switch (prop.type) {
      case "string":
        field = prop.enum
          ? z.enum(prop.enum as [string, ...string[]])
          : z.string()
        break
      case "number":
      case "integer":
        field = z.number()
        break
      case "boolean":
        field = z.boolean()
        break
      case "array":
        field = z.array(prop.items ? jsonSchemaToZod(prop.items) : z.any())
        break
      case "object":
        // If no properties defined, allow arbitrary keys (passthrough)
        if (!prop.properties) {
          field = z.record(z.string(), z.any())
        } else {
          field = jsonSchemaToZod(prop)
        }
        break
      default:
        field = z.any()
    }

    if (prop.description) {
      field = (field as any).describe(prop.description)
    }

    if (!required.has(key)) {
      field = field.optional() as any
    }

    shape[key] = field
  }

  return z.object(shape)
}

// -- Build in-process MCP server from ATN tool definitions --

interface ATNToolDef {
  name: string
  description: string
  input_schema: Record<string, any>
}

function buildATNMcpServer(toolDefs: ATNToolDef[]) {
  const tools = toolDefs.map((td) => {
    // Convert JSON Schema to Zod shape for the SDK
    const zodSchema = jsonSchemaToZod(td.input_schema)

    // The SDK tool() wants a raw shape object, not a ZodObject.
    // Extract the shape if it's a ZodObject, otherwise wrap in passthrough.
    let schemaArg: any
    if (zodSchema instanceof z.ZodObject) {
      schemaArg = zodSchema.shape
    } else {
      schemaArg = { _input: z.any().describe("Tool input") }
    }

    return tool(
      td.name,
      td.description,
      schemaArg,
      async (args: any) => {
        // Relay the tool call to Python and wait for the result
        const result = await relayToolCall(td.name, args)
        return {
          content: [{
            type: "text" as const,
            text: typeof result === "string" ? result : JSON.stringify(result),
          }],
        }
      },
    )
  })

  return createSdkMcpServer({
    name: "atn",
    version: "1.0.0",
    tools,
  })
}

// -- Session types --

interface BridgeSession {
  id: string
  query: Query
  iterator: AsyncIterator<SDKMessage, void>
  inputQueue: AsyncQueue<SDKUserMessage>
  resolvedModel: string | null
  createdAt: number
}

// -- Session Manager --

class SessionManager {
  private sessions = new Map<string, BridgeSession>()

  createAndSend(
    firstMessage: string,
    systemPrompt: string,
    options: Options,
  ): BridgeSession {
    const id = randomUUID()
    const inputQueue = new AsyncQueue<SDKUserMessage>()

    let content = ""
    if (systemPrompt) {
      content += systemPrompt + "\n\n"
    }
    content += `Human: ${firstMessage}`

    const firstUserMessage: SDKUserMessage = {
      type: "user",
      message: { role: "user", content },
      parent_tool_use_id: null,
      session_id: "",
    }

    inputQueue.push(firstUserMessage)

    const q = query({
      prompt: inputQueue,
      options,
    })

    const iterator = q[Symbol.asyncIterator]()

    const session: BridgeSession = {
      id,
      query: q,
      iterator,
      inputQueue,
      resolvedModel: null,
      createdAt: Date.now(),
    }

    this.sessions.set(id, session)
    log("session.created", { sessionId: id })
    return session
  }

  async collectResponse(session: BridgeSession): Promise<{
    text: string
    thinking: string[]
    toolCalls: Array<{ id: string; name: string; input: unknown }>
    usage: {
      input_tokens: number
      output_tokens: number
      cache_read_input_tokens: number
      cache_creation_input_tokens: number
    }
    model: string | null
    stopReason: string
  }> {
    let text = ""
    const thinking: string[] = []
    const toolCalls: Array<{ id: string; name: string; input: unknown }> = []
    let stopReason = "end_turn"

    try {
      while (true) {
        let message: SDKMessage | undefined
        try {
          const { value, done } = await session.iterator.next()
          if (done || !value) break
          message = value
        } catch (iterErr) {
          log("session.iterator.next() threw", {
            sessionId: session.id,
            error: iterErr instanceof Error ? iterErr.message : String(iterErr),
          })
          if (text) {
            text += "\n\n[Bridge: SDK error during execution - partial response returned]"
          }
          break
        }

        log("session.message", {
          sessionId: session.id,
          type: message.type,
          subtype: (message as any).subtype,
        })

        // Capture resolved model from SDK init message
        if (message.type === "system" && (message as any).subtype === "init") {
          const initModel = (message as any).model
          if (initModel && typeof initModel === "string") {
            session.resolvedModel = initModel
            log("session.resolvedModel", { sessionId: session.id, model: initModel })
          }
        }

        if (message.type === "assistant") {
          // Capture model from underlying API response
          if (!session.resolvedModel && (message as any).message?.model) {
            session.resolvedModel = (message as any).message.model
          }
          for (const block of message.message.content) {
            if (block.type === "text") {
              text += block.text
              if (block.text) {
                emitEvent({ type: "text_delta", text: block.text })
              }
            } else if (block.type === "thinking") {
              const thinkText = (block as any).thinking || ""
              if (thinkText) {
                thinking.push(thinkText)
                emitEvent({ type: "thinking", text: thinkText })
              }
            } else if (block.type === "tool_use") {
              toolCalls.push({
                id: block.id,
                name: block.name,
                input: block.input,
              })
            }
          }
        }

        if (message.type === "result") {
          const resultMsg = message as any

          // Extract usage from modelUsage
          let inputTokens = 0
          let outputTokens = 0
          let cacheReadTokens = 0
          let cacheCreationTokens = 0
          const modelUsage = resultMsg.modelUsage as Record<string, any> | undefined
          if (modelUsage) {
            for (const mu of Object.values(modelUsage)) {
              if (mu && typeof mu === "object") {
                inputTokens += mu.inputTokens ?? 0
                outputTokens += mu.outputTokens ?? 0
                cacheReadTokens += mu.cacheReadInputTokens ?? 0
                cacheCreationTokens += mu.cacheCreationInputTokens ?? 0
              }
            }
          }

          log("session.result", {
            sessionId: session.id,
            subtype: resultMsg.subtype,
            textLen: text.length,
            thinkingBlocks: thinking.length,
            toolCalls: toolCalls.length,
            usage: { inputTokens, outputTokens, cacheReadTokens, cacheCreationTokens },
          })

          emitEvent({ type: "done" })

          if (toolCalls.length > 0) {
            stopReason = "tool_use"
          }

          return {
            text,
            thinking,
            toolCalls,
            usage: {
              input_tokens: inputTokens,
              output_tokens: outputTokens,
              cache_read_input_tokens: cacheReadTokens,
              cache_creation_input_tokens: cacheCreationTokens,
            },
            model: session.resolvedModel,
            stopReason,
          }
        }
      }
    } catch (err) {
      log("collectResponse error", {
        error: err instanceof Error ? err.message : String(err),
      })
    }

    return {
      text,
      thinking,
      toolCalls,
      usage: { input_tokens: 0, output_tokens: 0, cache_read_input_tokens: 0, cache_creation_input_tokens: 0 },
      model: session.resolvedModel,
      stopReason,
    }
  }

  delete(id: string): boolean {
    const session = this.sessions.get(id)
    if (!session) return false

    try {
      session.inputQueue.done()
      session.query.close()
    } catch {
      // Already closed
    }
    this.sessions.delete(id)
    log("session.deleted", { sessionId: id })
    return true
  }

  destroy(): void {
    for (const id of [...this.sessions.keys()]) {
      this.delete(id)
    }
  }
}

// -- Protocol types --

interface CreateRequest {
  id: string
  type: "create"
  message: string
  system?: string
  model?: string
}

interface OrchestrateRequest {
  id: string
  type: "orchestrate"
  message: string
  system?: string
  system_prompt?: string   // Passed as SDK systemPrompt option (cached)
  model?: string
  tools: ATNToolDef[]
  max_turns?: number
  session_id?: string      // Resume this SDK session (loads history)
}

interface DeleteRequest {
  id: string
  type: "delete"
  session_id: string
}

interface ShutdownRequest {
  id: string
  type: "shutdown"
}

interface PingRequest {
  id: string
  type: "ping"
}

interface ToolResultMessage {
  type: "tool_result"
  call_id: string
  result: any
}

interface InterruptMessage {
  type: "interrupt"
}

interface UserMessageMessage {
  type: "user_message"
  content: string
}

interface SessionContextRequest {
  id: string
  type: "session_context"
  session_id: string
}

type BridgeRequest = CreateRequest | OrchestrateRequest | DeleteRequest | ShutdownRequest | PingRequest | SessionContextRequest

interface BridgeResponse {
  id: string
  ok: boolean
  session_id?: string
  text?: string
  thinking?: string[]
  tool_calls?: Array<{ id: string; name: string; input: unknown }>
  stop_reason?: string
  usage?: {
    input_tokens: number
    output_tokens: number
    cache_read_input_tokens: number
    cache_creation_input_tokens: number
  }
  context?: {
    num_turns: number
    total_cost_usd: number
    context_window: number
    max_output_tokens: number
  }
  model?: string
  error?: string
  // session_context response
  messages?: Array<{ role: string; content: string }>
  stats?: Record<string, unknown>
}

// -- Response writer --

function respond(resp: BridgeResponse): void {
  process.stdout.write(JSON.stringify(resp) + "\n")
}

// -- Main --

const sessionManager = new SessionManager()

// Active orchestration state — allows stdin to reach the running query
let activeOrchestration: {
  query: Query
  inputQueue: AsyncQueue<SDKUserMessage>
} | null = null

async function handleOrchestrateRequest(req: OrchestrateRequest): Promise<void> {
  try {
    const model = mapModelToClaudeModel(req.model || "sonnet")
    const maxTurns = req.max_turns || 20

    log("request.orchestrate", {
      model,
      maxTurns,
      toolCount: req.tools.length,
      messageLen: req.message.length,
    })

    // Build in-process MCP server with ATN tools
    const atnServer = buildATNMcpServer(req.tools)

    // Build tool allow-list for auto-approval
    const allowedTools = req.tools.map(t => `mcp__atn__${t.name}`)

    // Use streaming input (async generator) — required for MCP tools.
    // Queue stays open so mid-turn user messages and interrupt can be injected.
    const inputQueue = new AsyncQueue<SDKUserMessage>()

    // The message is now just the user's input — no system prompt or history
    // prepended.  System prompt goes via SDK option (cached).  History comes
    // from the SDK session via resume.
    inputQueue.push({
      type: "user",
      message: { role: "user", content: req.message },
      parent_tool_use_id: null,
      session_id: "",
    })

    // NOTE: We do NOT call inputQueue.done() here — the queue stays open
    // so that user_message and interrupt can be injected from stdin.

    // Build SDK options — systemPrompt and resume enable caching and
    // session continuity respectively.
    const sdkOptions: any = {
      maxTurns,
      model,
      pathToClaudeCodeExecutable: claudeExecutable,
      permissionMode: "bypassPermissions",
      allowDangerouslySkipPermissions: true,
      mcpServers: {
        atn: atnServer,
      },
      allowedTools,
      settingSources: [],
    }

    // Enable 1M context window for Sonnet models (reduces compaction frequency)
    if (model && model.toLowerCase().includes("sonnet")) {
      sdkOptions.betas = ["context-1m-2025-08-07"]
      log("request.orchestrate.beta", { betas: sdkOptions.betas })
    }

    // System prompt — passed as SDK option so it gets server-side caching.
    // Falls back to legacy inline system field for backward compat.
    const sysPrompt = req.system_prompt || req.system
    if (sysPrompt) {
      sdkOptions.systemPrompt = sysPrompt
    }

    // Session continuity — resume a previous session so the SDK loads
    // the conversation history and benefits from prompt caching.
    if (req.session_id) {
      sdkOptions.resume = req.session_id
    }

    log("request.orchestrate.options", {
      hasSystemPrompt: !!sysPrompt,
      resumeSession: req.session_id || null,
    })

    const q = query({
      prompt: inputQueue,
      options: sdkOptions,
    })

    // Expose to stdin handler for interrupt / user_message
    activeOrchestration = { query: q, inputQueue }

    // Collect the full response
    let text = ""
    const thinking: string[] = []
    let resolvedModel: string | null = null
    let inputTokens = 0
    let outputTokens = 0
    let cacheReadTokens = 0
    let cacheCreationTokens = 0
    let wasInterrupted = false
    let numTurns = 0
    let totalCostUsd = 0
    let contextWindow = 0
    let maxOutputTokens = 0
    let sessionId = ""

    try {
      for await (const message of q) {
        log("orchestrate.message", {
          type: message.type,
          subtype: (message as any).subtype,
        })

        if (message.type === "system" && (message as any).subtype === "init") {
          const initModel = (message as any).model
          if (initModel && typeof initModel === "string") {
            resolvedModel = initModel
          }
        }

        // Detect interrupt result
        if (message.type === "system" && (message as any).subtype === "interrupt") {
          wasInterrupted = true
          log("orchestrate.interrupted")
        }

        // Detect compaction status
        if (message.type === "system" && (message as any).subtype === "status") {
          const status = (message as any).status
          log("orchestrate.status", { status })
          emitEvent({ type: "status", status: status ?? "idle" })
        }

        // Detect compaction boundary
        if (message.type === "system" && (message as any).subtype === "compact_boundary") {
          const meta = (message as any).compact_metadata ?? {}
          log("orchestrate.compact_boundary", meta)
          emitEvent({
            type: "compaction",
            trigger: meta.trigger ?? "auto",
            pre_tokens: meta.pre_tokens ?? 0,
          })
        }

        if (message.type === "assistant") {
          if (!resolvedModel && (message as any).message?.model) {
            resolvedModel = (message as any).message.model
          }
          for (const block of message.message.content) {
            if (block.type === "text" && block.text) {
              text += block.text
              emitEvent({ type: "text_delta", text: block.text })
            } else if (block.type === "thinking") {
              const thinkText = (block as any).thinking || ""
              if (thinkText) {
                thinking.push(thinkText)
                emitEvent({ type: "thinking", text: thinkText })
              }
            } else if (block.type === "tool_use") {
              // Emit tool_use_start for SDK built-in tool calls (Read, Write, Bash, etc.)
              // These are handled internally by the SDK and don't go through relayToolCall.
              emitEvent({
                type: "tool_use_start",
                tool_use_id: (block as any).id || "",
                tool_name: (block as any).name || "",
                input: (block as any).input || {},
              })
            }
          }
        }

        // Tool use summary — SDK reports a text summary after tool calls complete.
        // The message has: { summary, preceding_tool_use_ids, uuid, session_id }
        if (message.type === "tool_use_summary") {
          const summaryMsg = message as any
          const toolUseIds: string[] = summaryMsg.preceding_tool_use_ids || []
          const summary: string = summaryMsg.summary || ""
          for (const toolUseId of toolUseIds) {
            emitEvent({
              type: "tool_use_result",
              tool_use_id: toolUseId,
              tool_name: "",  // not available in summary message
              is_error: false,
              result_preview: summary.slice(0, 500),
            })
          }
        }

        if (message.type === "result") {
          const resultMsg = message as any
          numTurns = resultMsg.num_turns ?? 0
          totalCostUsd = resultMsg.total_cost_usd ?? 0
          sessionId = resultMsg.session_id ?? ""

          const modelUsage = resultMsg.modelUsage as Record<string, any> | undefined
          if (modelUsage) {
            for (const mu of Object.values(modelUsage)) {
              if (mu && typeof mu === "object") {
                inputTokens += mu.inputTokens ?? 0
                outputTokens += mu.outputTokens ?? 0
                cacheReadTokens += mu.cacheReadInputTokens ?? 0
                cacheCreationTokens += mu.cacheCreationInputTokens ?? 0
                // Take the largest context window (in case multiple models used)
                if ((mu.contextWindow ?? 0) > contextWindow) {
                  contextWindow = mu.contextWindow ?? 0
                }
                if ((mu.maxOutputTokens ?? 0) > maxOutputTokens) {
                  maxOutputTokens = mu.maxOutputTokens ?? 0
                }
              }
            }
          }

          // Fallback: if SDK didn't populate contextWindow, derive from model name
          if (contextWindow === 0 && resolvedModel) {
            const m = resolvedModel.toLowerCase()
            if (m.includes("claude")) {
              // Sonnet with 1M beta gets 1M, otherwise 200k
              contextWindow = m.includes("sonnet") ? 1_000_000 : 200_000
              maxOutputTokens = maxOutputTokens || (m.includes("opus") ? 32_000 : 16_000)
            }
          }

          log("orchestrate.result", {
            textLen: text.length,
            thinkingBlocks: thinking.length,
            interrupted: wasInterrupted,
            numTurns,
            sessionId,
            usage: { inputTokens, outputTokens, cacheReadTokens, cacheCreationTokens },
            contextWindow,
          })

          // Close the input queue — this turn is done.
          // If a user_message was already pushed before this result, the SDK
          // will have consumed it.  Any future user_messages should start a
          // new orchestration, not inject into a completed one.
          try { inputQueue.done() } catch {}
          break
        }
      }
    } catch (err) {
      log("orchestrate.error", {
        error: err instanceof Error ? err.message : String(err),
      })
      if (!text) {
        text = `[Bridge: orchestration error: ${err instanceof Error ? err.message : String(err)}]`
      }
    } finally {
      // Clean up — close the input queue and clear active orchestration
      try { inputQueue.done() } catch {}
      activeOrchestration = null
    }

    emitEvent({ type: "done" })

    respond({
      id: req.id,
      ok: true,
      session_id: sessionId || undefined,
      text,
      thinking: thinking.length > 0 ? thinking : undefined,
      stop_reason: wasInterrupted ? "interrupted" : "end_turn",
      usage: {
        input_tokens: inputTokens,
        output_tokens: outputTokens,
        cache_read_input_tokens: cacheReadTokens,
        cache_creation_input_tokens: cacheCreationTokens,
      },
      context: {
        num_turns: numTurns,
        total_cost_usd: totalCostUsd,
        context_window: contextWindow,
        max_output_tokens: maxOutputTokens,
      },
      model: resolvedModel || undefined,
    })
  } catch (e) {
    respond({
      id: req.id,
      ok: false,
      error: e instanceof Error ? e.message : String(e),
    })
  }
}

async function handleRequest(req: BridgeRequest): Promise<void> {
  switch (req.type) {
    case "create": {
      try {
        const model = mapModelToClaudeModel(req.model || "sonnet")

        log("request.create", { model, messageLen: req.message.length })

        const session = sessionManager.createAndSend(req.message, req.system || "", {
          maxTurns: 1,
          model,
          pathToClaudeCodeExecutable: claudeExecutable,
          permissionMode: "bypassPermissions",
          allowDangerouslySkipPermissions: true,
        })

        const { text, thinking, toolCalls, usage, model: resolvedModel, stopReason } =
          await sessionManager.collectResponse(session)

        sessionManager.delete(session.id)

        respond({
          id: req.id,
          ok: true,
          session_id: session.id,
          text: text || "",
          thinking: thinking.length > 0 ? thinking : undefined,
          tool_calls: toolCalls.length > 0 ? toolCalls : undefined,
          stop_reason: stopReason,
          usage,
          model: resolvedModel || undefined,
        })
      } catch (e) {
        respond({
          id: req.id,
          ok: false,
          error: e instanceof Error ? e.message : String(e),
        })
      }
      break
    }

    case "orchestrate": {
      await handleOrchestrateRequest(req)
      break
    }

    case "delete": {
      const deleted = sessionManager.delete(req.session_id)
      respond({ id: req.id, ok: true })
      break
    }

    case "ping": {
      respond({ id: req.id, ok: true, type: "pong" })
      break
    }

    case "session_context": {
      try {
        const sessionId = (req as SessionContextRequest).session_id
        if (!sessionId) {
          respond({ id: req.id, ok: false, error: "Missing session_id" })
          break
        }

        log("request.session_context", { sessionId })
        const rawMessages = await getSessionMessages(sessionId)

        // Extract text content from each message
        const messages = rawMessages.map((m) => {
          let content = ""
          const msg = m.message as any
          if (typeof msg === "string") {
            content = msg
          } else if (msg?.content) {
            if (typeof msg.content === "string") {
              content = msg.content
            } else if (Array.isArray(msg.content)) {
              content = msg.content
                .filter((b: any) => b?.type === "text" && b?.text)
                .map((b: any) => b.text)
                .join("")
            }
          }
          return { role: m.type, content }
        })

        respond({
          id: req.id,
          ok: true,
          messages,
          stats: {
            total_messages: rawMessages.length,
            session_id: sessionId,
          },
        })
      } catch (e) {
        respond({
          id: req.id,
          ok: false,
          error: e instanceof Error ? e.message : String(e),
        })
      }
      break
    }

    case "shutdown": {
      log("shutdown requested")
      sessionManager.destroy()
      respond({ id: req.id, ok: true })
      setTimeout(() => process.exit(0), 100)
      break
    }

    default: {
      respond({
        id: (req as any).id || "unknown",
        ok: false,
        error: `Unknown request type: ${(req as any).type}`,
      })
    }
  }
}

// -- stdin line reader --
// Routes tool_result, interrupt, and user_message to active orchestration,
// everything else to handleRequest.

const rl = createInterface({ input: process.stdin })

rl.on("line", async (line: string) => {
  const trimmed = line.trim()
  if (!trimmed) return

  try {
    const parsed = JSON.parse(trimmed)

    // Route tool_result messages to the pending tool call
    if (parsed.type === "tool_result" && parsed.call_id) {
      handleToolResult(parsed.call_id, parsed.result)
      return
    }

    // Interrupt the active orchestration
    if (parsed.type === "interrupt") {
      if (activeOrchestration) {
        log("interrupt requested")
        try {
          await activeOrchestration.query.interrupt()
        } catch (err) {
          log("interrupt error", {
            error: err instanceof Error ? err.message : String(err),
          })
        }
      } else {
        log("interrupt requested but no active orchestration")
      }
      return
    }

    // Push a user message into the active orchestration
    if (parsed.type === "user_message" && parsed.content) {
      if (activeOrchestration) {
        log("user_message injected", { contentLen: parsed.content.length })
        activeOrchestration.inputQueue.push({
          type: "user",
          message: { role: "user", content: parsed.content },
          parent_tool_use_id: null,
          session_id: "",
        })
      } else {
        log("user_message received but no active orchestration")
      }
      return
    }

    // Everything else is a request
    await handleRequest(parsed as BridgeRequest)
  } catch (e) {
    respond({
      id: "parse-error",
      ok: false,
      error: `Failed to parse request: ${e instanceof Error ? e.message : String(e)}`,
    })
  }
})

rl.on("close", () => {
  log("stdin closed, shutting down")
  sessionManager.destroy()
  process.exit(0)
})

log("bridge ready", { pid: process.pid })
