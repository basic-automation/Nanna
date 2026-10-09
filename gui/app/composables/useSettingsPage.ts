/**
 * Shared state for the Settings page and its per-tab components.
 *
 * The page calls `provideSettingsPage()` once; each tab component calls
 * `useSettingsPage()` to access shared state (settings, toast, model catalog,
 * memory stats). Tab-exclusive state lives inside each tab component, which
 * registers an `onSettingsLoaded` hook so its state is (re)loaded every time
 * `loadSettings()` runs — matching the original monolithic behavior where a
 * single `loadSettings()` repopulated everything.
 */
import { computed, inject, provide, ref, type InjectionKey } from 'vue'
import { invoke } from '@tauri-apps/api/core'
import type { ModelOption } from '~/components/ModelPriorityList.vue'
import { isProviderAvailable } from '~/lib/providerAvailability'

export interface ToolInfo {
  name: string
  description: string
  enabled: boolean
}

export interface ExtendedSettings {
  anthropic_key_set: boolean
  openai_key_set: boolean
  openrouter_key_set: boolean
  github_key_set: boolean
  claude_proxy_enabled: boolean
  claude_proxy_url: string
  brave_key_set: boolean
  // Anthropic OAuth
  anthropic_oauth_logged_in: boolean
  anthropic_use_oauth: boolean
  provider: string
  available_providers: string[]
  model: string
  embedding_provider: string
  embedding_model: string
  embedding_enabled: boolean
  ollama_host: string
  /** Whether an Ollama bearer token is saved, and the server it is sent to.
   *  The token itself never reaches the page. */
  ollama_token_saved: boolean
  ollama_token_host: string | null
  /** `OLLAMA_API_KEY` is set: its token goes to whatever address is
   *  configured, ahead of any saved one. */
  ollama_token_from_env: boolean
  tools: ToolInfo[]
  auto_remember_messages: boolean
  max_compression_ratio: number
  min_remaining_memories: number
  scheduler_enabled: boolean
  heartbeat_enabled: boolean
  heartbeat_interval_seconds: number
  // No `thinking_enabled`: thinking is always on (owner directive 2026-08-04)
  // and the backend has no such field or command any more.
  streaming_enabled?: boolean
  max_tokens?: number
  agent_name?: string
  personality_mode?: string
  agent_max_iterations?: number | null
  agent_nudge_after_iterations?: number
  agent_nudge_interval_iterations?: number
}

export interface CognitiveMemoryStats {
  total_memories: number
  active: number
  dormant: number
  silent: number
  unavailable: number
}

export interface ModelInfo {
  id: string
  name: string
}

export interface OllamaModelInfo {
  name: string
  size_mb: number
  is_embedding_model: boolean
}

// Vision-capable models for OCR
const KNOWN_VISION_MODEL_PATTERNS = [
  'llava', 'deepseek-ocr', 'minicpm-v', 'moondream', 'bakllava',
  'cogvlm', 'internvl', 'qwen-vl', 'phi-3-vision', 'phi3v',
  'gpt-4o', 'gpt-4-vision', 'gpt-4-turbo',
  'claude-3', 'claude-opus', 'claude-sonnet', 'claude-haiku',
]

function isVisionCapable(modelId: string, provider: string): boolean {
  const id = modelId.toLowerCase()
  if (provider === 'anthropic') {
    // claude-3 and above support vision
    return id.includes('claude-3') || id.includes('claude-opus') ||
           id.includes('claude-sonnet') || id.includes('claude-haiku')
  }
  if (provider === 'openai') {
    return id.includes('gpt-4o') || id.includes('gpt-4-vision') || id.includes('gpt-4-turbo')
  }
  if (provider === 'ollama') {
    return KNOWN_VISION_MODEL_PATTERNS.some(p => id.includes(p))
  }
  // openrouter / github / claude-proxy: pattern match
  return KNOWN_VISION_MODEL_PATTERNS.some(p => id.includes(p))
}

function createSettingsPageStore() {
  const settings = ref<ExtendedSettings | null>(null)
  const toast = ref<{ message: string; type: 'success' | 'error' } | null>(null)
  /** Power-user knobs; toggled on settings page */
  const showAdvanced = ref(false)
  const memoryStats = ref<CognitiveMemoryStats | null>(null)

  // ── Model catalog ──
  const ollamaModels = ref<OllamaModelInfo[]>([])
  const loadingOllamaModels = ref(false)
  const ollamaStatus = ref<'unchecked' | 'checking' | 'connected' | 'error'>('unchecked')
  const ollamaError = ref('')
  const loadingModels = ref(false)

  // Dynamically fetched models from APIs
  const anthropicModels = ref<ModelInfo[]>([])
  const openaiModels = ref<ModelInfo[]>([])
  const openrouterModels = ref<ModelInfo[]>([])
  const openrouterEmbeddingModels = ref<ModelInfo[]>([])
  const githubModels = ref<ModelInfo[]>([])
  const claudeProxyModels = ref<ModelInfo[]>([])
  const claudeProxyHealthy = ref(false)

  /**
   * Providers the DAEMON's router can route to right now (null = daemon not
   * queried / unreachable). The daemon is authoritative for picker gating:
   * offering a model the daemon can't route means every chat turn dies with
   * "No provider available", so GUI-local login state only gates as a
   * fallback when this is null. See isProviderAvailable().
   */
  const daemonProviders = ref<string[] | null>(null)

  async function refreshDaemonProviders() {
    try {
      daemonProviders.value = await invoke<string[]>('get_daemon_providers')
    } catch (e) {
      console.warn('Daemon provider list unavailable, gating on local credentials:', e)
      daemonProviders.value = null
    }
  }

  const anthropicAvailable = computed(() => isProviderAvailable(
    daemonProviders.value,
    'anthropic',
    Boolean(settings.value?.anthropic_key_set || settings.value?.anthropic_oauth_logged_in),
  ))
  const openaiAvailable = computed(() => isProviderAvailable(
    daemonProviders.value,
    'openai',
    Boolean(settings.value?.openai_key_set),
  ))
  const openrouterAvailable = computed(() => isProviderAvailable(
    daemonProviders.value,
    'openrouter',
    Boolean(settings.value?.openrouter_key_set),
  ))
  const githubAvailable = computed(() => isProviderAvailable(
    daemonProviders.value,
    'github',
    Boolean(settings.value?.github_key_set),
  ))

  const allChatModels = computed<ModelOption[]>(() => {
    const models: ModelOption[] = []

    // Anthropic models (dynamically fetched from API)
    if (anthropicAvailable.value && anthropicModels.value.length > 0) {
      for (const m of anthropicModels.value) {
        models.push({ id: m.id, name: m.name, provider: 'anthropic', available: true })
      }
    }

    // OpenAI models (dynamically fetched from API)
    if (openaiAvailable.value && openaiModels.value.length > 0) {
      const chatModels = openaiModels.value.filter(m =>
        m.id.startsWith('gpt-') || m.id.startsWith('o1') || m.id.startsWith('o3') || m.id.startsWith('chatgpt')
      )
      for (const m of chatModels) {
        models.push({ id: m.id, name: m.name, provider: 'openai', available: true })
      }
    }

    // OpenRouter models (dynamically fetched from API)
    // Prefix with openrouter/ so parse_model_id recognizes the provider
    if (openrouterAvailable.value && openrouterModels.value.length > 0) {
      for (const m of openrouterModels.value) {
        models.push({ id: `openrouter/${m.id}`, name: m.name, provider: 'openrouter', available: true })
      }
    }

    // GitHub Models (dynamically fetched from API)
    if (githubAvailable.value && githubModels.value.length > 0) {
      for (const m of githubModels.value) {
        models.push({ id: `github/${m.id}`, name: m.name, provider: 'github', available: true })
      }
    }

    // Claude Proxy models (via claude-max-api-proxy)
    if (settings.value?.claude_proxy_enabled && claudeProxyModels.value.length > 0) {
      for (const m of claudeProxyModels.value) {
        models.push({ id: `claude-proxy/${m.id}`, name: `${m.name} (Proxy)`, provider: 'claude-proxy', available: claudeProxyHealthy.value })
      }
    }

    // Ollama models (dynamically fetched - always dynamic)
    for (const m of ollamaModels.value.filter(m => !m.is_embedding_model)) {
      models.push({ id: `ollama/${m.name}`, name: m.name, provider: 'ollama', available: ollamaStatus.value === 'connected' })
    }

    return models
  })

  const allEmbeddingModels = computed<ModelOption[]>(() => {
    const models: ModelOption[] = []

    // Ollama embedding models (local, free — listed first)
    for (const m of ollamaModels.value.filter(m => m.is_embedding_model)) {
      models.push({ id: `ollama/${m.name}`, name: `${m.name} (${m.size_mb}MB, local)`, provider: 'ollama', available: ollamaStatus.value === 'connected' })
    }

    // OpenAI embedding models
    if (openaiAvailable.value && openaiModels.value.length > 0) {
      const embeddingModels = openaiModels.value.filter(m => m.id.startsWith('text-embedding'))
      for (const m of embeddingModels) {
        models.push({ id: `openai/${m.id}`, name: m.name, provider: 'openai', available: true })
      }
    }

    // OpenRouter embedding models (from dedicated embeddings endpoint)
    if (openrouterAvailable.value && openrouterEmbeddingModels.value.length > 0) {
      for (const m of openrouterEmbeddingModels.value) {
        models.push({ id: `openrouter/${m.id}`, name: `${m.name} (OpenRouter)`, provider: 'openrouter', available: true })
      }
    }

    // GitHub embedding models
    if (githubAvailable.value && githubModels.value.length > 0) {
      const embeddingModels = githubModels.value.filter(m =>
        m.id.includes('embed') || m.id.includes('embedding')
      )
      for (const m of embeddingModels) {
        models.push({ id: `github/${m.id}`, name: m.name, provider: 'github', available: true })
      }
    }

    return models
  })

  // Models available for context summarization (any chat model can be used)
  const allSummarizationModels = computed<ModelOption[]>(() => {
    const models: ModelOption[] = []

    // Ollama chat models (listed first - local, free, private)
    for (const m of ollamaModels.value.filter(m => !m.is_embedding_model)) {
      models.push({ id: `ollama/${m.name}`, name: `${m.name} (local)`, provider: 'ollama', available: ollamaStatus.value === 'connected' })
    }

    // Anthropic models
    if (anthropicAvailable.value && anthropicModels.value.length > 0) {
      for (const m of anthropicModels.value) {
        models.push({ id: `anthropic/${m.id}`, name: m.name, provider: 'anthropic', available: true })
      }
    }

    // OpenAI models
    if (openaiAvailable.value && openaiModels.value.length > 0) {
      const chatModels = openaiModels.value.filter(m =>
        m.id.startsWith('gpt-') || m.id.startsWith('o1') || m.id.startsWith('o3') || m.id.startsWith('chatgpt')
      )
      for (const m of chatModels) {
        models.push({ id: `openai/${m.id}`, name: m.name, provider: 'openai', available: true })
      }
    }

    // OpenRouter models
    if (openrouterAvailable.value && openrouterModels.value.length > 0) {
      for (const m of openrouterModels.value) {
        models.push({ id: `openrouter/${m.id}`, name: m.name, provider: 'openrouter', available: true })
      }
    }

    // GitHub Models
    if (githubAvailable.value && githubModels.value.length > 0) {
      for (const m of githubModels.value) {
        models.push({ id: `github/${m.id}`, name: m.name, provider: 'github', available: true })
      }
    }

    return models
  })

  const allOcrModels = computed<ModelOption[]>(() => {
    const models: ModelOption[] = []

    // Anthropic claude-3+ (vision capable)
    if (anthropicAvailable.value && anthropicModels.value.length > 0) {
      for (const m of anthropicModels.value) {
        if (isVisionCapable(m.id, 'anthropic')) {
          models.push({ id: m.id, name: m.name, provider: 'anthropic', available: true })
        }
      }
    }

    // OpenAI vision models
    if (openaiAvailable.value && openaiModels.value.length > 0) {
      for (const m of openaiModels.value) {
        if (isVisionCapable(m.id, 'openai')) {
          models.push({ id: m.id, name: m.name, provider: 'openai', available: true })
        }
      }
    }

    // Ollama vision models (local — listed after cloud for cost efficiency awareness)
    for (const m of ollamaModels.value.filter(m => !m.is_embedding_model)) {
      if (isVisionCapable(m.name, 'ollama')) {
        models.push({
          id: `ollama/${m.name}`,
          name: `${m.name} (local)`,
          provider: 'ollama',
          available: ollamaStatus.value === 'connected',
        })
      }
    }

    // OpenRouter vision models
    if (openrouterAvailable.value && openrouterModels.value.length > 0) {
      for (const m of openrouterModels.value) {
        if (isVisionCapable(m.id, 'openrouter')) {
          models.push({ id: `openrouter/${m.id}`, name: m.name, provider: 'openrouter', available: true })
        }
      }
    }

    // GitHub vision models
    if (githubAvailable.value && githubModels.value.length > 0) {
      for (const m of githubModels.value) {
        if (isVisionCapable(m.id, 'github')) {
          models.push({ id: `github/${m.id}`, name: m.name, provider: 'github', available: true })
        }
      }
    }

    return models
  })

  // Available models for routing (all chat models as select options)
  const routingModelOptions = computed(() => {
    return allChatModels.value
      .filter(m => m.available)
      .map(m => ({
        value: m.id,
        label: `${m.name} (${m.provider})`,
      }))
  })

  // ── Settings loading ──

  // Tab components register hooks here so their exclusive state is
  // (re)populated every time loadSettings() runs.
  const settingsLoadedHooks: Array<() => Promise<void> | void> = []
  function onSettingsLoaded(hook: () => Promise<void> | void) {
    settingsLoadedHooks.push(hook)
  }

  async function loadSettings() {
    try {
      settings.value = await invoke<ExtendedSettings>('get_extended_settings')

      // The daemon's live provider set gates the model pickers; re-query it on
      // every settings (re)load so a login/logout is reflected immediately.
      await refreshDaemonProviders()

      // Let each tab repopulate its own state from the fresh settings
      for (const hook of settingsLoadedHooks) {
        await hook()
      }

      // Always refresh Ollama models to populate the lists
      await refreshOllamaModels()
      await refreshModels()
    } catch (e) {
      console.error('Failed to load settings:', e)
      showToast('Failed to load settings', 'error')
    }
  }

  async function refreshModels() {
    if (!settings.value) return
    loadingModels.value = true

    // Fetch models from all available providers in parallel
    const promises: Promise<void>[] = []

    // Anthropic models (fetch if API key OR OAuth is available)
    if (settings.value.anthropic_key_set || settings.value.anthropic_oauth_logged_in) {
      promises.push(
        invoke<ModelInfo[]>('get_anthropic_models')
          .then(models => { anthropicModels.value = models })
          .catch(e => {
            console.warn('Failed to fetch Anthropic models:', e)
            anthropicModels.value = [] // Will use fallback in computed
          })
      )
    }

    // OpenAI models
    if (settings.value.openai_key_set) {
      promises.push(
        invoke<ModelInfo[]>('get_openai_models')
          .then(models => { openaiModels.value = models })
          .catch(e => {
            console.warn('Failed to fetch OpenAI models:', e)
            openaiModels.value = []
          })
      )
    }

    // OpenRouter models
    if (settings.value.openrouter_key_set) {
      promises.push(
        invoke<ModelInfo[]>('get_openrouter_models')
          .then(models => { openrouterModels.value = models })
          .catch(e => {
            console.warn('Failed to fetch OpenRouter models:', e)
            openrouterModels.value = []
          })
      )
      promises.push(
        invoke<ModelInfo[]>('get_openrouter_embedding_models')
          .then(models => { openrouterEmbeddingModels.value = models })
          .catch(e => {
            console.warn('Failed to fetch OpenRouter embedding models:', e)
            openrouterEmbeddingModels.value = []
          })
      )
    }

    // GitHub models
    if (settings.value.github_key_set) {
      promises.push(
        invoke<ModelInfo[]>('get_github_models')
          .then(models => { githubModels.value = models })
          .catch(e => {
            console.warn('Failed to fetch GitHub models:', e)
            githubModels.value = []
          })
      )
    }

    // Claude Proxy models (check health first)
    if (settings.value.claude_proxy_enabled) {
      promises.push(
        invoke<boolean>('check_claude_proxy_health')
          .then(async (healthy) => {
            claudeProxyHealthy.value = healthy
            if (healthy) {
              try {
                claudeProxyModels.value = await invoke<ModelInfo[]>('get_claude_proxy_models')
              } catch (e) {
                console.warn('Failed to fetch Claude Proxy models:', e)
                claudeProxyModels.value = []
              }
            } else {
              claudeProxyModels.value = []
            }
          })
          .catch(e => {
            console.warn('Failed to check Claude Proxy health:', e)
            claudeProxyHealthy.value = false
            claudeProxyModels.value = []
          })
      )
    }

    // Wait for all fetches to complete
    await Promise.all(promises)
    loadingModels.value = false
  }

  async function refreshOllamaModels() {
    loadingOllamaModels.value = true
    ollamaStatus.value = 'checking'
    ollamaError.value = ''
    try {
      ollamaModels.value = await invoke<OllamaModelInfo[]>('get_ollama_models')
      ollamaStatus.value = 'connected'
    } catch (e: any) {
      ollamaStatus.value = 'error'
      ollamaError.value = e.message || String(e)
      ollamaModels.value = []
    } finally {
      loadingOllamaModels.value = false
    }
  }

  async function loadMemoryStats() {
    try {
      memoryStats.value = await invoke<CognitiveMemoryStats>('get_cognitive_memory_stats')
    } catch (e) {
      console.error('Failed to load memory stats:', e)
    }
  }

  async function saveApiKey(provider: string, apiKey: string) {
    try {
      await invoke('set_provider_api_key', { provider, apiKey })
      showToast(`${formatProvider(provider)} API key saved`, 'success')
      await loadSettings()
    } catch (e: any) {
      showToast(`Couldn't complete that: ${e.message || e}`, 'error')
    }
  }

  function formatProvider(provider: string): string {
    const names: Record<string, string> = {
      anthropic: 'Anthropic', openai: 'OpenAI', openrouter: 'OpenRouter', github: 'GitHub Models', 'claude-proxy': 'Claude Proxy', ollama: 'Ollama',
    }
    return names[provider] || provider
  }

  function showToast(message: string, type: 'success' | 'error') {
    toast.value = { message, type }
    setTimeout(() => { toast.value = null }, 3000)
  }

  return {
    settings,
    toast,
    showAdvanced,
    memoryStats,
    ollamaModels,
    loadingOllamaModels,
    ollamaStatus,
    ollamaError,
    loadingModels,
    anthropicModels,
    openaiModels,
    openrouterModels,
    openrouterEmbeddingModels,
    githubModels,
    claudeProxyModels,
    claudeProxyHealthy,
    daemonProviders,
    refreshDaemonProviders,
    anthropicAvailable,
    openaiAvailable,
    openrouterAvailable,
    githubAvailable,
    allChatModels,
    allEmbeddingModels,
    allSummarizationModels,
    allOcrModels,
    routingModelOptions,
    onSettingsLoaded,
    loadSettings,
    refreshModels,
    refreshOllamaModels,
    loadMemoryStats,
    saveApiKey,
    formatProvider,
    showToast,
  }
}

export type SettingsPageStore = ReturnType<typeof createSettingsPageStore>

const SettingsPageKey: InjectionKey<SettingsPageStore> = Symbol('settings-page')

/** Called once by the settings page to create and provide the shared store. */
export function provideSettingsPage(): SettingsPageStore {
  const store = createSettingsPageStore()
  provide(SettingsPageKey, store)
  return store
}

/** Called by settings tab components to access the shared store. */
export function useSettingsPage(): SettingsPageStore {
  const store = inject(SettingsPageKey)
  if (!store) {
    throw new Error('useSettingsPage() must be used within the settings page')
  }
  return store
}
