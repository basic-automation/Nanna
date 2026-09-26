/**
 * Install a hermetic Tauri 2 runtime shim for browser Playwright tests.
 *
 * Real `@tauri-apps/api` packages read `window.__TAURI_INTERNALS__` for invoke /
 * transformCallback, and `window.__TAURI_EVENT_PLUGIN_INTERNALS__` for event
 * unregistration. This fixture supplies those internals so the stock Nuxt app
 * can boot without a desktop shell or live daemon.
 *
 * Deterministic + offline: chat streaming is simulated, settings keys live in
 * the in-memory MockState, no network or keyring is touched.
 */
import type { Page } from '@playwright/test'
import { type MockOptions } from './mock-state'

export type { MockOptions }

declare global {
  interface Window {
    __NANNA_E2E__?: {
      emit: (event: string, payload: unknown) => void
      setBackendStatus: (status: Record<string, unknown>) => void
      /** The daemon answers: connected, running, every boot field cleared. */
      attach: () => void
      getState: () => unknown
      runStream?: (sessionId: string, reply?: string) => Promise<void>
      forceCrash?: () => void
    }
    __TAURI_INTERNALS__?: Record<string, unknown>
    __TAURI_EVENT_PLUGIN_INTERNALS__?: Record<string, unknown>
    isTauri?: boolean
  }
}

/**
 * Browser-side install body. Playwright serializes this function via toString()
 * and injects it as an init script — keep it free of outer-scope closures.
 * Options arrive as the serialized argument.
 */
function installInPage(options = {}) {

  if (window.__NANNA_E2E__) return;

  // ─── Mock state (mirrors mock-state.ts) ───────────────────────────────
  function nowIso() { return new Date().toISOString(); }
  function uid(prefix) {
    return prefix + '-' + Math.random().toString(36).slice(2, 10) + Date.now().toString(36);
  }
  const apiKeySet = options.apiKeySet !== false;
  const boot = options.boot || null;
  const connected = !boot && options.backendConnected !== false;
  const bootState = boot ? (boot.daemon_state || 'starting') : null;
  const sessions = Array.isArray(options.sessions)
    ? options.sessions.map((s) => ({
        id: s.id || uid('sess'),
        name: s.name || 'Chat',
        created_at: s.created_at || nowIso(),
        updated_at: s.updated_at || nowIso(),
        message_count: s.message_count ?? 0,
        workspace_id: s.workspace_id ?? null,
        workspace_name: s.workspace_name ?? null,
      }))
    : [{
        id: 'sess-seed-1',
        name: 'Welcome',
        created_at: nowIso(),
        updated_at: nowIso(),
        message_count: 0,
        workspace_id: null,
        workspace_name: null,
      }];
  const state = {
    backend: {
      mode: boot ? 'disconnected' : 'daemon',
      connected,
      daemon_url: connected ? 'ws://127.0.0.1:5149' : null,
      daemon_state: boot ? bootState : (connected ? 'running' : 'stopped'),
      version: '0.1.0-e2e',
      message: connected ? null : 'Daemon not reachable on 5149 (e2e mock)',
      starting_for_s: boot
        ? (boot.starting_for_s !== undefined ? boot.starting_for_s : (bootState === 'starting' ? 0 : null))
        : null,
      retrying: boot ? Boolean(boot.retrying) : false,
      init_in_progress: boot
        ? (typeof boot.init_in_progress === 'boolean' ? boot.init_in_progress : bootState === 'starting')
        : false,
      last_error: boot ? (boot.last_error || null) : null,
    },
    // The sidecar's output for get_boot_log: this spawn's lines, oldest first.
    bootLog: boot && Array.isArray(boot.log)
      ? boot.log.map((l) => ({ ...l }))
      : [{ stream: 'stdout', line: 'Daemon ready (e2e mock)' }],
    restartCalls: 0,
    config: {
      theme: 'palenight',
      model: 'mock-model',
      api_key_set: apiKeySet,
      available_models: ['mock-model', 'local-tiny'],
      provider: 'local',
      streaming_enabled: true,
      agent_name: 'Nanna',
      max_tokens: 4096,
      anthropic_api_key: apiKeySet ? 'sk-ant-e2e-mock' : '',
      openai_api_key: '',
      openrouter_api_key: '',
      ollama_host: 'http://localhost:11434',
      // The server a saved Ollama token is for; null = no token saved.
      ollama_token_host: null,
    },
    sessions: sessions.slice(),
    messages: Object.assign(Object.create(null), options.messages || {}),
    logs: [
      { timestamp: nowIso(), level: 'info', target: 'nanna_daemon', message: 'Daemon ready (e2e mock)', source: 'daemon' },
      { timestamp: nowIso(), level: 'debug', target: 'nanna_gui', message: 'Attached over IPC', source: 'embedded' },
      { timestamp: nowIso(), level: 'info', target: 'nanna_gui', message: 'boot complete', source: 'embedded' },
    ],
    logClearedBefore: null,
    memories: [
      { id: 'mem-1', content: 'User prefers calm replies', importance: 0.8, created_at: nowIso(), updated_at: nowIso(), category: 'preference', workspace_id: null },
    ],
    tools: [
      { name: 'read_file', description: 'Read a file', category: 'files', enabled: true, source: 'default' },
      { name: 'exec', description: 'Run a shell command', category: 'shell', enabled: true, source: 'default' },
      { name: 'web_search', description: 'Search the web', category: 'web', enabled: false, source: 'default' },
    ],
    workspaces: [],
    channels: [
      { name: 'telegram', configured: false, connected: false, status: 'not_configured' },
      { name: 'discord', configured: false, connected: false, status: 'not_configured' },
    ],
    cronJobs: [],
    modelStats: { models: [], total_requests: 0, total_tokens: 0, total_cost_usd: 0, costs: [] },
    toolStats: { tools: [], total_calls: 0 },
    agentStats: { agents: [], clusters: [] },
    systemPrompt: 'You are Nanna.',
    streamingEnabled: true,
    streamAuto: options.streamAuto !== false,
    streamChunks: options.streamChunks || ['Hello from ', 'the mock ', 'assistant.'],
    callbacks: Object.create(null),
    nextCallbackId: 1,
    listeners: Object.create(null),
    nextEventId: 1,
  };

  function camel(obj) {
    if (!obj || typeof obj !== 'object') return obj;
    const out = Object.create(null);
    for (const [k, v] of Object.entries(obj)) {
      const ck = k.replace(/_([a-z])/g, (_, c) => c.toUpperCase());
      out[ck] = v;
      out[k] = v;
    }
    return out;
  }
  function pick(args, ...keys) {
    if (!args) return undefined;
    for (const k of keys) {
      if (args[k] !== undefined) return args[k];
      const snake = k.replace(/[A-Z]/g, (m) => '_' + m.toLowerCase());
      if (args[snake] !== undefined) return args[snake];
    }
    return undefined;
  }
  function getSession(id) {
    return state.sessions.find((s) => s.id === id) || null;
  }
  function ensureMessages(id) {
    if (!state.messages[id]) state.messages[id] = [];
    return state.messages[id];
  }

  // ─── Event bus via transformCallback ──────────────────────────────────
  function transformCallback(callback, once) {
    const id = state.nextCallbackId++;
    const wrapped = (payload) => {
      try { callback(payload); } catch (err) { console.error('[e2e-mock] callback error', err); }
      if (once) delete state.callbacks[id];
    };
    state.callbacks[id] = wrapped;
    return id;
  }
  function unregisterCallback(id) { delete state.callbacks[id]; }
  function runCallback(id, payload) {
    const cb = state.callbacks[id];
    if (cb) cb(payload);
  }
  function emitEvent(event, payload) {
    const bucket = state.listeners[event] || [];
    for (const entry of bucket.slice()) {
      runCallback(entry.handlerId, { event, id: entry.eventId, payload });
    }
  }
  function registerListener(event, handlerId) {
    const eventId = state.nextEventId++;
    if (!state.listeners[event]) state.listeners[event] = [];
    state.listeners[event].push({ eventId, handlerId });
    return eventId;
  }
  function unregisterListener(event, eventId) {
    const bucket = state.listeners[event];
    if (!bucket) return;
    state.listeners[event] = bucket.filter((e) => e.eventId !== eventId);
  }

  async function runStream(sessionId, replyText) {
    if (!state.streamingSessions) state.streamingSessions = Object.create(null);
    // Mark active so cancel can stop mid-stream.
    const token = { cancelled: false };
    state.streamingSessions[sessionId] = token;

    const chunks = replyText
      ? String(replyText).match(/.{1,12}(\s|$)/g) || [String(replyText)]
      : state.streamChunks.slice();
    let accumulated = '';
    for (const chunk of chunks) {
      if (token.cancelled) break;
      accumulated += chunk;
      emitEvent('stream-chunk', { session_id: sessionId, chunk, done: false });
      // Slow enough that the Stop button is interactable in e2e.
      await new Promise((r) => setTimeout(r, 80));
    }
    const msgs = ensureMessages(sessionId);
    if (token.cancelled) {
      // Seal partial work, matching daemon cancel semantics.
      if (accumulated) {
        msgs.push({ role: 'assistant', content: accumulated + '\n\n_(Stopped by user)_', timestamp: nowIso() });
      } else {
        msgs.push({ role: 'assistant', content: '_(Stopped by user)_', timestamp: nowIso() });
      }
    } else {
      msgs.push({ role: 'assistant', content: accumulated, timestamp: nowIso() });
    }
    const sess = getSession(sessionId);
    if (sess) {
      sess.message_count = msgs.length;
      sess.updated_at = nowIso();
    }
    emitEvent('stream-chunk', {
      session_id: sessionId,
      chunk: '',
      done: true,
      message: token.cancelled
        ? { role: 'assistant', content: accumulated + (accumulated ? '\n\n_(Stopped by user)_' : '_(Stopped by user)_') }
        : { role: 'assistant', content: accumulated },
    });
    delete state.streamingSessions[sessionId];
  }

  // ─── Command handlers ─────────────────────────────────────────────────
  async function handleCommand(cmd, rawArgs) {
    const args = camel(rawArgs || {});

    // Event plugin
    if (cmd === 'plugin:event|listen') {
      const event = args.event;
      const handlerId = args.handler;
      return registerListener(event, handlerId);
    }
    if (cmd === 'plugin:event|unlisten') {
      unregisterListener(args.event, args.eventId);
      return null;
    }
    if (cmd === 'plugin:event|emit' || cmd === 'plugin:event|emit_to') {
      emitEvent(args.event, args.payload);
      return null;
    }

    // Window plugin — no-ops that keep getCurrentWindow happy
    if (typeof cmd === 'string' && cmd.startsWith('plugin:window|')) {
      if (cmd === 'plugin:window|get_all_windows') return [{ label: 'main' }];
      return null;
    }
    if (typeof cmd === 'string' && cmd.startsWith('plugin:webview|')) return null;
    if (typeof cmd === 'string' && cmd.startsWith('plugin:app|')) {
      if (cmd === 'plugin:app|version') return '0.1.0-e2e';
      if (cmd === 'plugin:app|name') return 'Nanna';
      return null;
    }
    if (typeof cmd === 'string' && cmd.startsWith('plugin:notification|')) return true;
    if (typeof cmd === 'string' && cmd.startsWith('plugin:dialog|')) return null;
    if (typeof cmd === 'string' && cmd.startsWith('plugin:shell|')) return null;
    if (typeof cmd === 'string' && cmd.startsWith('plugin:')) return null;

    switch (cmd) {
      case 'init_backend': {
        // Backend::init joins the init already in flight and answers when it
        // ends (backend.rs checks every 100 ms). It never attaches by itself
        // here; a spec says when the daemon answers, with attach().
        while (!state.backend.connected
          && (state.backend.daemon_state === 'starting' || state.backend.init_in_progress === true)) {
          await new Promise((r) => setTimeout(r, 100));
        }
        return state.backend.connected ? 'daemon' : 'disconnected';
      }
      case 'get_backend_status':
        return { ...state.backend };
      case 'get_boot_log':
        return state.bootLog.map((l) => ({ ...l }));
      case 'restart_daemon':
        // The old boot is stopped and a fresh init starts in the background;
        // the command returns at once and the status shows the new boot.
        state.restartCalls += 1;
        Object.assign(state.backend, {
          mode: 'disconnected',
          connected: false,
          daemon_url: null,
          daemon_state: 'starting',
          starting_for_s: 0,
          retrying: false,
          init_in_progress: true,
        });
        state.bootLog = [{ stream: 'stdout', line: 'Starting the daemon (e2e mock restart)' }];
        return null;

      case 'list_sessions':
        return state.sessions.map((s) => ({ ...s }));
      case 'create_session': {
        const session = {
          id: uid('sess'),
          name: pick(args, 'name') || 'New chat',
          created_at: nowIso(),
          updated_at: nowIso(),
          message_count: 0,
          workspace_id: pick(args, 'workspaceId', 'workspace_id') ?? null,
          workspace_name: null,
        };
        state.sessions.unshift(session);
        state.messages[session.id] = [];
        // keep session counts coherent for any consumer reading backend metadata
        return { ...session };
      }
      case 'delete_session': {
        const id = pick(args, 'sessionId', 'id', 'session_id');
        state.sessions = state.sessions.filter((s) => s.id !== id);
        delete state.messages[id];
        return true;
      }
      case 'rename_session': {
        const id = pick(args, 'sessionId', 'id', 'session_id');
        const name = pick(args, 'name') || 'Renamed';
        const sess = getSession(id);
        if (sess) {
          sess.name = name;
          sess.updated_at = nowIso();
          emitEvent('session-renamed', { id, name });
        }
        return sess ? { ...sess } : null;
      }
      case 'clear_all_sessions':
        state.sessions = [];
        state.messages = Object.create(null);
        emitEvent('sessions-cleared', {});
        return true;
      case 'get_session_history': {
        const id = pick(args, 'sessionId', 'id', 'session_id');
        return ensureMessages(id).map((m) => ({ ...m }));
      }
      case 'get_session_run_state': {
        const sid = pick(args, 'sessionId', 'session_id', 'id');
        const running = Boolean(state.streamingSessions && state.streamingSessions[sid]);
        return {
          is_running: running,
          accumulated_text: '',
          accumulated_thinking: '',
          active_tools: [],
          queued_count: 0,
        };
      }

      case 'send_message': {
        const sessionId = pick(args, 'sessionId', 'session_id');
        const message = pick(args, 'message') || '';
        if (!sessionId) throw new Error('sessionId required');
        if (!state.backend.connected) throw new Error('Daemon not reachable on 5149');
        const msgs = ensureMessages(sessionId);
        msgs.push({ role: 'user', content: message, timestamp: nowIso() });
        const sess = getSession(sessionId);
        if (sess) {
          sess.message_count = msgs.length;
          sess.updated_at = nowIso();
          if (sess.name === 'New chat' || sess.name === 'Welcome') {
            sess.name = message.slice(0, 32) || sess.name;
          }
        }
        if (state.streamAuto) {
          queueMicrotask(() => { void runStream(sessionId); });
        }
        return { ok: true };
      }
      case 'cancel_session':
      case 'cancel_agent': {
        const id = pick(args, 'sessionId', 'session_id', 'id');
        if (state.streamingSessions && id && state.streamingSessions[id]) {
          state.streamingSessions[id].cancelled = true;
        } else if (state.streamingSessions) {
          // cancel all if id missing
          for (const k of Object.keys(state.streamingSessions)) {
            state.streamingSessions[k].cancelled = true;
          }
        }
        return true;
      }

      case 'get_config':
        return { ...state.config };
      case 'save_config': {
        const cfg = pick(args, 'config') || args;
        Object.assign(state.config, cfg || {});
        if (typeof state.config.anthropic_api_key === 'string') {
          state.config.api_key_set = state.config.anthropic_api_key.length > 0
            || !!(state.config.openai_api_key)
            || !!(state.config.openrouter_api_key);
        }
        return { ...state.config };
      }
      case 'set_provider_api_key': {
        const provider = String(pick(args, 'provider') || 'anthropic').toLowerCase();
        const key = pick(args, 'apiKey', 'api_key', 'key') || '';
        if (provider.includes('openai')) state.config.openai_api_key = key;
        else if (provider.includes('openrouter')) state.config.openrouter_api_key = key;
        else state.config.anthropic_api_key = key;
        state.config.api_key_set = !!(
          state.config.anthropic_api_key || state.config.openai_api_key || state.config.openrouter_api_key
        );
        return true;
      }
      case 'set_provider':
        state.config.provider = pick(args, 'provider') || state.config.provider;
        return true;
      case 'set_model':
        state.config.model = pick(args, 'model') || state.config.model;
        return true;
      case 'set_streaming_enabled':
        state.config.streaming_enabled = !!(pick(args, 'enabled') ?? true);
        return true;
      case 'set_agent_name':
        state.config.agent_name = pick(args, 'name') || state.config.agent_name;
        return true;
      case 'set_max_tokens':
        state.config.max_tokens = Number(pick(args, 'maxTokens', 'max_tokens') || 4096);
        return true;
      case 'export_config':
        return JSON.stringify(state.config, null, 2);
      case 'import_config':
        return true;
      case 'get_extended_settings':
        return {
          anthropic_key_set: !!(state.config.anthropic_api_key),
          openai_key_set: !!(state.config.openai_api_key),
          openrouter_key_set: !!(state.config.openrouter_api_key),
          github_key_set: false,
          claude_proxy_enabled: false,
          claude_proxy_url: '',
          brave_key_set: false,
          anthropic_oauth_logged_in: false,
          anthropic_use_oauth: false,
          provider: state.config.provider,
          available_providers: ['anthropic', 'openai', 'openrouter', 'ollama', 'local'],
          model: state.config.model,
          streaming_enabled: state.config.streaming_enabled,
          agent_name: state.config.agent_name,
          max_tokens: state.config.max_tokens,
          api_key_set: state.config.api_key_set,
          ollama_host: state.config.ollama_host,
          // Whether an Ollama token is saved and for which server — the
          // real command never sends the token itself.
          ollama_token_saved: state.config.ollama_token_host !== null,
          ollama_token_host: state.config.ollama_token_host,
          ollama_token_from_env: false,
        };

      case 'get_system_prompt':
        return state.systemPrompt;
      case 'set_system_prompt':
        state.systemPrompt = pick(args, 'prompt', 'systemPrompt') || state.systemPrompt;
        return true;

      case 'get_daemon_logs': {
        const limit = Number(pick(args, 'limit') || 500);
        let entries = state.logs.slice();
        if (state.logClearedBefore) {
          const cut = Date.parse(state.logClearedBefore);
          entries = entries.filter((e) => Date.parse(e.timestamp) > cut);
        }
        return entries.slice(-limit);
      }

      case 'list_memories':
        return state.memories.map((m) => ({ ...m }));
      case 'delete_memory':
        state.memories = state.memories.filter((m) => m.id !== pick(args, 'id', 'memoryId'));
        return true;
      case 'update_memory': {
        const id = pick(args, 'id', 'memoryId');
        const mem = state.memories.find((m) => m.id === id);
        if (mem) Object.assign(mem, pick(args, 'memory') || args);
        return mem || null;
      }
      case 'clear_memories':
      case 'clear_all_memories':
        state.memories = [];
        return true;
      case 'get_cognitive_memory_stats':
        return { total: state.memories.length, by_category: {}, avg_importance: 0.5 };
      case 'set_max_compression_ratio':
      case 'set_min_remaining_memories':
      case 'trigger_consolidation':
        return true;

      case 'list_tools':
        return state.tools.map((t) => ({ ...t }));
      case 'get_tool':
      case 'get_user_tool':
        return state.tools.find((t) => t.name === pick(args, 'name', 'toolName')) || null;
      case 'get_tool_source':
        return '// mock tool source\nexport default { name: "mock" }\n';
      case 'create_user_tool':
      case 'update_user_tool':
      case 'delete_user_tool':
      case 'test_skill':
        return true;
      case 'get_tool_stats':
        return { ...state.toolStats };
      case 'get_tool_stats_daily':
      case 'get_tool_stats_hourly':
        return [];
      case 'get_tool_call_log':
        return [];

      case 'list_workspaces':
        return state.workspaces.map((w) => ({ ...w }));
      case 'open_workspace':
      case 'close_workspace':
      case 'init_workspace':
      case 'reload_workspace':
      case 'set_active_workspace':
      case 'clear_active_workspace':
        return true;
      case 'check_workspace_validity':
        return { valid: true, markers: ['README.md'], issues: [] };

      case 'get_channel_status':
      case 'get_enhanced_channel_status':
        return state.channels.map((c) => ({ ...c }));
      case 'save_channel_config':
      case 'test_channel_connection':
      case 'subscribe_channel_status':
      case 'unsubscribe_channel_status':
        return true;

      case 'list_cron_jobs':
        return state.cronJobs.slice();
      case 'create_cron_job':
      case 'update_cron_job':
      case 'delete_cron_job':
      case 'run_cron_job_now':
      case 'set_cron_job_enabled':
      case 'set_scheduler_enabled':
      case 'set_heartbeat_enabled':
      case 'set_heartbeat_interval':
        return true;
      case 'get_cron_job_history':
        return [];
      case 'validate_cron_expression':
        return { valid: true, next: nowIso() };

      case 'list_tasks':
        return { tasks: [] };

      case 'get_model_stats':
        return { ...state.modelStats };
      case 'get_global_stats':
        return { requests: 0, tokens: 0, cost_usd: 0 };
      case 'get_agent_stats':
        return { ...state.agentStats };
      case 'get_agent_clusters':
        return [];
      case 'subscribe_agent_events':
        return true;

      case 'get_model_status':
        return {
          active_model: state.config.model || 'mock-model',
          fallback_reason: null,
          rate_limited_models: [],
          provider: state.config.provider,
          model: state.config.model,
          healthy: true,
        };
      case 'get_model_routing':
        return { enabled: true, strategy: 'complexity' };
      case 'set_model_routing':
      case 'get_routing_first_turn_primary':
        return true;
      case 'set_routing_first_turn_primary':
        return true;
      case 'get_chat_model_priority':
      case 'get_embedding_model_priority':
      case 'get_summarization_model_priority':
      case 'get_ocr_model_priority':
        return [state.config.model];
      case 'set_chat_model_priority':
      case 'set_embedding_model_priority':
      case 'set_summarization_model_priority':
      case 'set_ocr_model_priority':
      case 'set_embedding_config':
      case 'set_sub_agent_models':
      case 'set_personality_mode':
      case 'set_agent_iteration_policy':
      case 'set_claude_proxy':
      case 'set_use_embedded_ocr':
        return true;
      case 'set_ollama_host':
        state.config.ollama_host = String(pick(args, 'host') || '').trim().replace(/\/+$/, '');
        return 'Ollama host saved: ' + state.config.ollama_host;
      case 'set_ollama_api_key': {
        // Saved for the configured server; blank removes it.
        const key = String(pick(args, 'key') || '').trim();
        state.config.ollama_token_host = key ? state.config.ollama_host : null;
        return key ? 'Ollama token saved' : 'Ollama token removed';
      }
      case 'get_sub_agent_models':
        return [];
      case 'get_use_embedded_ocr':
        return true;
      case 'get_anthropic_models':
      case 'get_openai_models':
      case 'get_openrouter_models':
      case 'get_openrouter_embedding_models':
      case 'get_claude_proxy_models':
      case 'get_github_models':
        return [
          { id: state.config.model, name: state.config.model },
          { id: 'gpt-4o-mini', name: 'GPT-4o mini' },
          { id: 'text-embedding-3-small', name: 'text-embedding-3-small' },
        ];
      case 'get_ollama_models':
        return [
          { name: 'qwen3.5:9b', size_mb: 6000, is_embedding_model: false },
          { name: 'nomic-embed-text', size_mb: 274, is_embedding_model: true },
        ];
      // The daemon's hardened probe, as the wizard sees it: a live local
      // server with every configured model pulled.
      case 'probe_ollama':
        return {
          base_url: 'http://localhost:11434',
          reachable: true,
          reason: null,
          models: [
            { name: 'qwen3.5:9b', size_mb: 6000, is_embedding_model: false },
            { name: 'nomic-embed-text:latest', size_mb: 274, is_embedding_model: true },
          ],
          wanted: ['qwen3.5:9b'],
          missing: [],
        };
      case 'check_claude_proxy_health':
        return false;
      case 'get_cognitive_memory_stats':
      case 'get_memory_stats':
        return {
          total_memories: state.memories?.length || 0,
          by_category: {},
          avg_importance: 0.5,
          last_dreamed_at: null,
          dream_count: 0,
        };
      case 'import_claude_code_credentials':
      case 'logout_anthropic_oauth':
      case 'save_anthropic_oauth_token':
      case 'run_claude_setup_token':
        return true;

      // Daemon-reported provider availability (the model pickers gate on this,
      // not on GUI-local key state). Mirror the real daemon: a provider exists
      // when its credential does; Ollama always registers.
      case 'get_daemon_providers':
        return [
          ...(state.config.anthropic_api_key ? ['anthropic'] : []),
          ...(state.config.openai_api_key ? ['openai'] : []),
          ...(state.config.openrouter_api_key ? ['openrouter'] : []),
          'ollama',
        ];

      case 'get_close_mode':
        return 'ask';
      case 'handle_window_close':
        // What the default close mode answers: the window opens CloseDialog.
        return 'ask';
      case 'set_close_mode':
      case 'hide_to_tray':
      case 'perform_quit':
        return true;

      default:
        console.warn('[e2e-mock] unhandled invoke:', cmd, args);
        return null;
    }
  }

  async function invoke(cmd, args, _options) {
    try {
      return await handleCommand(cmd, args);
    } catch (err) {
      console.error('[e2e-mock] invoke failed', cmd, err);
      throw err;
    }
  }

  // ─── Install Tauri 2 internals ────────────────────────────────────────
  window.__TAURI_INTERNALS__ = {
    invoke,
    transformCallback,
    unregisterCallback,
    convertFileSrc(filePath, protocol) {
      return (protocol || 'asset') + '://localhost/' + String(filePath || '').replace(/^\/+/, '');
    },
    metadata: {
      currentWindow: { label: 'main' },
      currentWebview: { label: 'main' },
    },
    plugins: {},
  };
  window.__TAURI_EVENT_PLUGIN_INTERNALS__ = {
    unregisterListener,
  };
  window.isTauri = true;
  window.__NANNA_E2E_READY__ = true;

  // IPC-style event delivery used by some Tauri builds (defensive)
  window.__TAURI_EVENT_PLUGIN_INTERNALS__.unregisterListener = unregisterListener;

  window.__NANNA_E2E__ = {
    emit: emitEvent,
    setBackendStatus(status) {
      Object.assign(state.backend, status || {});
      if (typeof status?.connected === 'boolean') {
        state.backend.connected = status.connected;
        state.backend.mode = 'daemon'; // mode stays daemon; connected flag drives DISCONNECTED label
        // An explicit daemon_state wins: "disconnected and starting again" is
        // what the health monitor's restart looks like.
        if (typeof status.daemon_state !== 'string') {
          state.backend.daemon_state = status.connected ? 'running' : 'stopped';
        }
        state.backend.daemon_url = status.connected ? (state.backend.daemon_url || 'ws://127.0.0.1:5149') : null;
        state.backend.message = status.connected
          ? null
          : (status.message || 'Daemon not reachable on 5149 (e2e mock)');
      }
    },
    attach() {
      Object.assign(state.backend, {
        mode: 'daemon',
        connected: true,
        daemon_url: 'ws://127.0.0.1:5149',
        daemon_state: 'running',
        message: null,
        starting_for_s: null,
        retrying: false,
        init_in_progress: false,
        last_error: null,
      });
    },
    getState() {
      return {
        backend: { ...state.backend },
        config: { ...state.config },
        sessions: state.sessions.map((s) => ({ ...s })),
        logCount: state.logs.length,
        api_key_set: state.config.api_key_set,
        restartCalls: state.restartCalls,
      };
    },
    runStream,
    clearLogs() {
      state.logClearedBefore = nowIso();
      state.logs.push({
        timestamp: nowIso(),
        level: 'info',
        target: 'nanna_gui',
        message: 'Logs cleared (e2e)',
        source: 'embedded',
      });
    },
    appendLog(entry) {
      state.logs.push(Object.assign({
        timestamp: nowIso(),
        level: 'info',
        target: 'e2e',
        message: 'log',
        source: 'daemon',
      }, entry || {}));
    },
    forceCrash() {
      queueMicrotask(() => {
        throw new Error('E2E forced crash');
      });
    },
  };

  console.info('[e2e-mock] Tauri internals installed', {
    apiKeySet: state.config.api_key_set,
    connected: state.backend.connected,
    sessions: state.sessions.length,
  });
}

export async function installTauriMock(page: Page, options: MockOptions = {}): Promise<void> {
  await page.addInitScript(installInPage, options ?? {})
}

export async function e2eEmit(page: Page, event: string, payload: unknown = {}): Promise<void> {
  await page.evaluate(
    ({ event: ev, payload: pl }) => {
      window.__NANNA_E2E__?.emit(ev, pl)
    },
    { event, payload },
  )
}

export async function e2eSetBackendStatus(
  page: Page,
  status: Record<string, unknown>,
): Promise<void> {
  await page.evaluate((s) => {
    window.__NANNA_E2E__?.setBackendStatus(s)
  }, status)
}

export async function e2eAttach(page: Page): Promise<void> {
  await page.evaluate(() => {
    window.__NANNA_E2E__?.attach()
  })
}

export async function e2eGetState(page: Page): Promise<unknown> {
  return page.evaluate(() => window.__NANNA_E2E__?.getState() ?? null)
}

export async function e2eForceCrash(page: Page): Promise<void> {
  await page.evaluate(() => {
    window.__NANNA_E2E__?.forceCrash?.()
  })
}
