<template>
  <div class="flex h-full">
    <!-- Sidebar: Tools List -->
    <aside class="w-64 border-r border-white/[0.04] flex flex-col">
      <header class="px-3 py-3 border-b border-white/[0.04]">
        <div class="flex items-center justify-between mb-2">
          <h2 class="font-semibold text-nanna-text text-sm">Tools</h2>
          <div class="flex gap-1">
            <button @click="refreshTools" class="p-1 rounded hover:bg-nanna-primary/20 text-nanna-text-muted hover:text-nanna-text transition-colors" title="Refresh">
              <RefreshCw class="w-4 h-4" :class="{ 'animate-spin': refreshing }" />
            </button>
            <button @click="openCreateModal" class="p-1 rounded hover:bg-nanna-primary/20 text-nanna-text-muted hover:text-nanna-text transition-colors" title="New Tool">
              <Plus class="w-4 h-4" />
            </button>
          </div>
        </div>
        <UiInput v-model="searchQuery" placeholder="Search tools..." size="sm" class="text-xs" />
      </header>

      <div class="flex-1 min-h-0 flex flex-col p-2">
        <PageState
          v-if="loading || !isOnline || loadError || filteredTools.length === 0"
          :state="loading ? 'loading' : (!isOnline ? 'offline' : (loadError ? 'error' : 'empty'))"
          :title="loading ? 'Loading tools…' : (!isOnline ? 'Daemon offline' : (loadError ? 'Could not load tools' : 'No tools found'))"
          :description="loading
            ? 'Asking the daemon for the skill registry.'
            : (!isOnline
              ? 'Tools are loaded by the daemon. Reconnect to browse or author skills.'
              : (loadError || 'No tools match the current filter.'))"
          :primary-action="loading ? '' : ((!isOnline || loadError) ? 'Retry' : 'Create tool')"
          :primary-busy="loading"
          @primary="onToolsPrimary"
        />

        <!-- Tools list -->
        <VirtualList
          v-else-if="filteredTools.length > 60"
          :items="filteredTools"
          :item-height="36"
          :overscan="8"
          class="flex-1 min-h-0"
        >
          <template #default="{ item: tool }">
            <div
              @click="selectTool(tool)"
              class="group flex items-center gap-2 px-2 py-1.5 rounded cursor-pointer transition-colors h-full"
              :class="selectedTool?.name === tool.name
                ? 'bg-nanna-accent/20 text-nanna-accent'
                : 'hover:bg-nanna-primary/10 text-nanna-text-muted hover:text-nanna-text'"
            >
              <Wrench class="w-4 h-4 flex-shrink-0" />
              <span class="text-sm truncate flex-1">{{ tool.name }}</span>
              <span @click.stop>
                <UiSwitch
                  :model-value="tool.enabled"
                  :disabled="togglingTools.has(tool.name)"
                  :label="`${tool.enabled ? 'Disable' : 'Enable'} ${tool.name}`"
                  @update:model-value="setToolEnabled(tool, $event)"
                />
              </span>
            </div>
          </template>
        </VirtualList>
        <div v-else class="space-y-1 flex-1 min-h-0 overflow-y-auto">
          <div
            v-for="tool in filteredTools"
            :key="tool.name"
            @click="selectTool(tool)"
            class="group flex items-center gap-2 px-2 py-1.5 rounded cursor-pointer transition-colors"
            :class="selectedTool?.name === tool.name
              ? 'bg-nanna-accent/20 text-nanna-accent'
              : 'hover:bg-nanna-primary/10 text-nanna-text-muted hover:text-nanna-text'"
          >
            <Wrench class="w-4 h-4 flex-shrink-0" />
            <span class="text-sm truncate flex-1">{{ tool.name }}</span>
            <span @click.stop>
              <UiSwitch
                :model-value="tool.enabled"
                :disabled="togglingTools.has(tool.name)"
                :label="`${tool.enabled ? 'Disable' : 'Enable'} ${tool.name}`"
                @update:model-value="setToolEnabled(tool, $event)"
              />
            </span>
          </div>
        </div>
      </div>

      <!-- Tool count -->
      <footer class="px-3 py-2 border-t border-white/[0.04] text-[10px] text-nanna-text-dim">
        {{ enabledCount }} of {{ tools.length }} tools enabled
      </footer>
    </aside>

    <!-- Main Content -->
    <main class="flex-1 flex flex-col min-w-0">
      <!-- No tool selected -->
      <div v-if="!selectedTool && !creating" class="flex-1 flex items-center justify-center">
        <div class="text-center max-w-md px-8">
          <Wrench class="w-16 h-16 mx-auto mb-4 text-nanna-text-dim" />
          <h3 class="text-lg font-semibold text-nanna-text mb-2">Tools</h3>
          <p class="text-sm text-nanna-text-muted mb-6">
            Tools extend Nanna's capabilities. Select a tool from the sidebar to view its details,
            or create a new custom tool.
          </p>
          <div class="flex gap-2 justify-center">
            <UiButton @click="openCreateModal" size="sm">
              <Plus class="w-4 h-4 mr-1" /> New Tool
            </UiButton>
            <UiButton @click="openTemplates" variant="ghost" size="sm">
              <FileCode2 class="w-4 h-4 mr-1" /> Templates
            </UiButton>
          </div>
          <McpServerList :servers="mcpServers" class="mt-8" />
        </div>
      </div>

      <!-- Tool Details / Editor -->
      <div v-else class="flex-1 flex flex-col min-h-0">
        <!-- Header -->
        <header class="px-4 py-2 border-b border-white/[0.04]">
          <div class="flex items-center justify-between">
            <div class="flex items-center gap-3">
              <div class="flex items-center gap-2">
                <Wrench class="w-5 h-5 text-nanna-accent" />
                <span class="font-mono font-semibold text-nanna-text">
                  {{ creating ? 'New Tool' : selectedTool?.name }}
                </span>
              </div>
              <UiBadge v-if="hasChanges" variant="warning" size="sm">unsaved</UiBadge>
              <UiBadge v-if="selectedTool?.is_user_tool" variant="secondary" size="sm">user tool</UiBadge>
            </div>
            <div class="flex items-center gap-2">
              <div v-if="selectedTool && !creating" class="flex items-center gap-2 mr-1">
                <span class="text-xs text-nanna-text-muted">
                  {{ selectedTool.enabled ? 'Enabled' : 'Disabled' }}
                </span>
                <UiSwitch
                  :model-value="selectedTool.enabled"
                  :disabled="togglingTools.has(selectedTool.name)"
                  :label="`${selectedTool.enabled ? 'Disable' : 'Enable'} ${selectedTool.name}`"
                  @update:model-value="setToolEnabled(selectedTool, $event)"
                />
              </div>
              <UiButton @click="testTool" variant="ghost" size="sm" :disabled="!canTest">
                <Play class="w-4 h-4 mr-1" /> Test
              </UiButton>
              <UiButton @click="saveTool" size="sm" :disabled="!hasChanges || saving" v-if="isEditable">
                <Save class="w-4 h-4 mr-1" />
                {{ saving ? 'Saving...' : 'Save' }}
              </UiButton>
              <UiButton v-if="selectedTool && !creating && isEditable" @click="confirmDelete" variant="ghost" size="sm" class="text-red-400 hover:text-red-300">
                <Trash2 class="w-4 h-4" />
              </UiButton>
            </div>
          </div>
        </header>

        <!-- Tabs -->
        <div class="flex px-3 py-1.5 border-b border-white/[0.04]">
          <UiGlassTabs v-model="activeTab" :tabs="editorTabs" size="sm" />
        </div>

        <!-- Content -->
        <div class="flex-1 flex min-h-0">
          <!-- Details Tab -->
          <div v-show="activeTab === 'details'" class="flex-1 p-4 overflow-y-auto">
            <div class="max-w-2xl space-y-6">
              <!-- Tool Info -->
              <div>
                <h3 class="text-sm font-semibold text-nanna-text mb-2">Description</h3>
                <p class="text-sm text-nanna-text-muted">
                  {{ selectedTool?.description || editingTool.description || 'No description' }}
                </p>
              </div>

              <!-- Parameters -->
              <div v-if="toolDetails?.parameters">
                <h3 class="text-sm font-semibold text-nanna-text mb-2">Parameters</h3>
                <div class="glass-well rounded-lg p-3 space-y-2">
                  <div v-for="(param, key) in toolDetails.parameters.properties || {}" :key="key" class="flex items-start gap-2 text-sm">
                    <code class="px-1.5 py-0.5 glass-tag rounded text-nanna-accent text-xs">{{ key }}</code>
                    <span class="text-nanna-text-dim">{{ param.type }}</span>
                    <span v-if="toolDetails.parameters.required?.includes(key)" class="text-red-400 text-xs">required</span>
                    <span v-if="param.description" class="text-nanna-text-muted text-xs flex-1">— {{ param.description }}</span>
                  </div>
                  <div v-if="!toolDetails.parameters.properties || Object.keys(toolDetails.parameters.properties).length === 0" class="text-sm text-nanna-text-dim">
                    No parameters
                  </div>
                </div>
              </div>

              <!-- Edit fields (for editable tools) -->
              <div v-if="isEditable || creating" class="space-y-4">
                <div>
                  <label class="block text-xs font-medium text-nanna-text-muted mb-1">Tool Name</label>
                  <UiInput v-model="editingTool.name" placeholder="my_tool" class="font-mono text-sm" :disabled="!creating" />
                </div>
                <div>
                  <label class="block text-xs font-medium text-nanna-text-muted mb-1">Description</label>
                  <UiTextarea v-model="editingTool.description" placeholder="What does this tool do?" :rows="2" class="text-sm" />
                </div>
                <div>
                  <label class="block text-xs font-medium text-nanna-text-muted mb-1">Output Routing</label>
                  <select v-model="editingTool.outputTarget" class="w-full px-3 py-2 glass-well rounded-lg text-sm text-nanna-text">
                    <option value="memory">Memory (default) — large results stored in memory, stubbed in context</option>
                    <option value="context">Context — results always inline, summarized if large</option>
                  </select>
                  <p class="text-[10px] text-nanna-text-dim mt-1">
                    Use "context" for tools whose results the agent needs to see directly (e.g. search, lookup).
                  </p>
                </div>
              </div>
            </div>
          </div>

          <!-- Code Tab -->
          <div v-show="activeTab === 'code'" class="flex-1 flex flex-col min-w-0">
            <!-- Creating: Name input -->
            <div v-if="creating" class="px-4 py-3 border-b border-white/[0.04]">
              <div class="flex gap-4">
                <div class="flex-1">
                  <label class="block text-xs font-medium text-nanna-text-muted mb-1">Tool Name</label>
                  <UiInput v-model="editingTool.name" placeholder="my_tool" class="font-mono text-sm" />
                </div>
                <div class="w-32">
                  <label class="block text-xs font-medium text-nanna-text-muted mb-1">Type</label>
                  <select v-model="editingTool.toolType" class="w-full px-3 py-2 glass-well rounded-lg text-sm text-nanna-text">
                    <option value="script">Script (JS/TS)</option>
                    <option value="manifest">Manifest (YAML)</option>
                  </select>
                </div>
              </div>
            </div>

            <!-- Loading source -->
            <div v-if="!creating && loadingSource" class="flex-1 flex items-center justify-center">
              <Loader2 class="w-5 h-5 animate-spin text-nanna-text-muted" />
            </div>

            <!-- No source available -->
            <div v-else-if="!creating && !editingTool.code" class="flex-1 flex items-center justify-center">
              <div class="text-center max-w-sm px-8">
                <Wrench class="w-10 h-10 mx-auto mb-3 text-nanna-text-dim" />
                <p class="text-sm text-nanna-text-muted mb-2">Source unavailable</p>
                <p class="text-xs text-nanna-text-dim">
                  Could not load source code for this tool.
                </p>
              </div>
            </div>

            <!-- Monaco Editor -->
            <div v-else class="flex-1 min-h-0">
              <VueMonacoEditor
                v-model:value="editingTool.code"
                :language="editorLanguage"
                theme="nanna-dark"
                :options="editorOptions"
                @mount="onEditorMount"
                class="h-full"
              />
            </div>
          </div>

          <!-- Test Panel (slide out) -->
          <aside v-if="showTestPanel" class="w-80 border-l border-white/[0.04] flex flex-col">
            <header class="px-3 py-2 border-b border-white/[0.04] flex items-center justify-between">
              <span class="text-sm font-semibold text-nanna-text">Test Tool</span>
              <button @click="showTestPanel = false" class="p-1 rounded hover:bg-nanna-primary/20">
                <X class="w-4 h-4 text-nanna-text-muted" />
              </button>
            </header>

            <div class="flex-1 overflow-y-auto p-3 space-y-4">
              <!-- Auto-generated form from schema -->
              <div v-if="toolDetails?.parameters?.properties">
                <div class="text-xs font-medium text-nanna-text-muted mb-2">Input Parameters</div>
                <div class="space-y-3">
                  <div v-for="(param, key) in toolDetails.parameters.properties" :key="key">
                    <label class="block text-xs text-nanna-text mb-1">
                      {{ key }}
                      <span v-if="toolDetails.parameters.required?.includes(key)" class="text-red-400">*</span>
                    </label>
                    <UiInput
                      v-model="testInputs[key]"
                      :placeholder="param.description || param.type"
                      size="sm"
                      class="text-xs"
                    />
                  </div>
                </div>
              </div>

              <!-- Raw JSON input -->
              <div v-else>
                <label class="block text-xs font-medium text-nanna-text-muted mb-1">Input (JSON)</label>
                <textarea
                  v-model="testInputJson"
                  class="w-full h-24 p-2 glass-well rounded text-xs font-mono text-nanna-text resize-none"
                  placeholder='{ "param": "value" }'
                ></textarea>
              </div>

              <UiButton @click="runTest" :disabled="testing" class="w-full" size="sm">
                <Play class="w-4 h-4 mr-1" />
                {{ testing ? 'Running...' : 'Run Test' }}
              </UiButton>

              <!-- Test Result -->
              <div v-if="testResult !== null" class="space-y-2">
                <div class="text-xs font-medium text-nanna-text-muted">Result</div>
                <div class="p-3 glass-well rounded text-xs font-mono whitespace-pre-wrap" :class="testError ? 'text-red-400' : 'text-green-400'">
                  {{ testResult }}
                </div>
                <div v-if="testDuration" class="text-[10px] text-nanna-text-dim">
                  Completed in {{ testDuration }}ms
                </div>
              </div>
            </div>
          </aside>
        </div>
      </div>
    </main>

    <!-- Create Tool Modal -->
    <UiModal v-model="showCreateModal" title="Create New Tool" size="lg">
      <div class="space-y-4">
        <p class="text-sm text-nanna-text-muted">
          Choose a template to get started:
        </p>

        <div class="grid grid-cols-2 gap-3">
          <button
            v-for="template in templates"
            :key="template.id"
            @click="createFromTemplate(template)"
            class="p-4 text-left glass-panel hover:bg-nanna-primary/10 hover:border-nanna-accent/50 rounded-lg transition-colors"
          >
            <div class="text-lg mb-1">{{ template.icon }}</div>
            <div class="font-medium text-nanna-text text-sm">{{ template.name }}</div>
            <div class="text-xs text-nanna-text-muted mt-1">{{ template.description }}</div>
          </button>
        </div>

        <div class="flex justify-end pt-4 border-t border-white/[0.04]">
          <UiButton @click="showCreateModal = false" variant="ghost">Cancel</UiButton>
        </div>
      </div>
    </UiModal>

    <!-- Delete Confirmation -->
    <UiModal v-model="showDeleteModal" title="Delete Tool">
      <p class="text-nanna-text-muted mb-4">
        Are you sure you want to delete <strong class="text-nanna-text">{{ selectedTool?.name }}</strong>?
        This action cannot be undone.
      </p>
      <div class="flex justify-end gap-2">
        <UiButton @click="showDeleteModal = false" variant="ghost">Cancel</UiButton>
        <UiButton @click="deleteTool" variant="destructive" :disabled="deleting">
          {{ deleting ? 'Deleting...' : 'Delete' }}
        </UiButton>
      </div>
    </UiModal>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted } from 'vue'
import { parseMcpServers, type McpServerState } from '~/lib/mcpServers'
import { invoke } from '@tauri-apps/api/core'
import { VueMonacoEditor } from '@guolao/vue-monaco-editor'
import {
  Plus, RefreshCw, Loader2, FileCode2, Wrench, Play, Save, Trash2, X
} from '@lucide/vue'

const { isOnline } = useBackend()
const toast = useToast()
const { confirm } = useConfirm()

interface Tool {
  name: string
  description: string
  enabled: boolean
  // Snake_case because that is what the daemon and the Tauri command emit
  // verbatim (no serde rename anywhere in the chain), matching how every other
  // daemon-sourced field is read in this app. It was previously declared as
  // `isUserTool`, which no payload has ever carried, so it read `undefined`.
  is_user_tool?: boolean
}

interface ToolDetails {
  name: string
  description: string
  parameters?: {
    type: string
    properties?: Record<string, { type: string; description?: string }>
    required?: string[]
  }
}

interface Template {
  id: string
  name: string
  icon: string
  description: string
  toolType: 'script' | 'manifest'
  code: string
}

// State
const loading = ref(true)
const loadError = ref<string | null>(null)
const refreshing = ref(false)
const saving = ref(false)
const deleting = ref(false)
const testing = ref(false)
const searchQuery = ref('')
const tools = ref<Tool[]>([])
/** Names with a toggle in flight, so a switch cannot be double-fired. */
const togglingTools = ref<Set<string>>(new Set())
const selectedTool = ref<Tool | null>(null)
const toolDetails = ref<ToolDetails | null>(null)
const creating = ref(false)
const hasChanges = ref(false)
const showTestPanel = ref(false)
const showCreateModal = ref(false)
const showDeleteModal = ref(false)
const loadingSource = ref(false)
const activeTab = ref<'details' | 'code'>('details')

// Editing state
const editingTool = ref({
  name: '',
  description: '',
  toolType: 'script' as 'script' | 'manifest',
  code: '',
  outputTarget: 'memory' as 'memory' | 'context',
})
const originalCode = ref('')

// Test state
const testInputs = ref<Record<string, string>>({})
const testInputJson = ref('{}')
const testResult = ref<string | null>(null)
const testError = ref(false)
const testDuration = ref<number | null>(null)

// Templates
const templates: Template[] = [
  {
    id: 'blank-ts',
    name: 'Blank TypeScript',
    icon: '📝',
    description: 'Empty tool template',
    toolType: 'script',
    code: `export default {
  name: 'my_tool',
  description: 'Describe what your tool does',
  parameters: {
    type: 'object',
    properties: {
      input: { type: 'string', description: 'Input value' }
    },
    required: ['input']
  },
  execute(params) {
    return \`Result: \${params.input}\`;
  }
}`
  },
  {
    id: 'api-fetch',
    name: 'API Fetch',
    icon: '🌐',
    description: 'Fetch data from an API',
    toolType: 'script',
    code: `export default {
  name: 'api_fetch',
  description: 'Fetch data from an API endpoint',
  parameters: {
    type: 'object',
    properties: {
      url: { type: 'string', description: 'API endpoint URL' },
      method: { type: 'string', enum: ['GET', 'POST'], default: 'GET' }
    },
    required: ['url']
  },
  async execute(params) {
    const response = await fetch(params.url, {
      method: params.method || 'GET'
    });
    return await response.json();
  }
}`
  },
  {
    id: 'file-processor',
    name: 'File Processor',
    icon: '📄',
    description: 'Read and process files',
    toolType: 'manifest',
    code: `name: file_processor
description: Process a file with a shell script

parameters:
  type: object
  properties:
    file:
      type: string
      description: Path to the input file
    operation:
      type: string
      enum: [count_lines, word_count, checksum]
  required: [file, operation]

shell: |
  case "{{operation}}" in
    count_lines) wc -l "{{file}}" ;;
    word_count) wc -w "{{file}}" ;;
    checksum) sha256sum "{{file}}" ;;
  esac`
  },
  {
    id: 'python-script',
    name: 'Python Script',
    icon: '🐍',
    description: 'Run Python code',
    toolType: 'manifest',
    code: `name: python_tool
description: Execute a Python script

parameters:
  type: object
  properties:
    data:
      type: string
      description: Input data to process
  required: [data]

python: |
  import json
  import sys

  data = sys.argv[1] if len(sys.argv) > 1 else ''

  # Your Python code here
  result = f"Processed: {data}"
  print(result)`
  }
]

// Computed
const filteredTools = computed(() => {
  const query = searchQuery.value.toLowerCase()
  return tools.value.filter(t => t.name.toLowerCase().includes(query))
})

/** A disabled tool is not available, so the footer must not count it as one. */
const enabledCount = computed(() => tools.value.filter(t => t.enabled).length)

const editorTabs = computed(() => [
  { id: 'details', label: 'Details' },
  { id: 'code', label: editingTool.value.toolType === 'manifest' ? 'Manifest' : 'Code' },
])

const editorLanguage = computed(() => {
  if (editingTool.value.toolType === 'manifest') return 'yaml'
  return 'typescript'
})

const editorOptions = computed(() => ({
  minimap: { enabled: false },
  lineNumbers: 'on',
  fontSize: 13,
  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
  fontLigatures: true,
  scrollBeyondLastLine: false,
  wordWrap: 'on',
  tabSize: 2,
  automaticLayout: true,
  readOnly: !isEditable.value && !creating.value,
}))

const canTest = computed(() => {
  return selectedTool.value !== null || (creating.value && editingTool.value.code.trim().length > 0)
})

const isEditable = computed(() => {
  // All tools are editable in principle - the backend will decide if it can actually save changes
  return true
})

// Methods
function onToolsPrimary() {
  if (!isOnline.value || loadError.value) {
    void refreshTools()
    return
  }
  openCreateModal()
}


async function refreshTools() {
  refreshing.value = true
  try {
    const result = await invoke<Tool[]>('list_tools')
    tools.value = result
  } catch (e) {
    console.error('Failed to load tools:', e)
  } finally {
    refreshing.value = false
    loading.value = false
  }
  await refreshMcpServers()
}

/** Configured MCP servers and why any of them is not serving tools. */
const mcpServers = ref<McpServerState[]>([])

async function refreshMcpServers() {
  try {
    mcpServers.value = parseMcpServers(await invoke<unknown>('get_mcp_servers'))
  } catch (e) {
    // An older daemon or a dropped connection: show no MCP section rather
    // than a stale one.
    mcpServers.value = []
    console.warn('Could not read MCP server state:', e)
  }
}

/**
 * Enable or disable a tool.
 *
 * Optimistic, then reverted if the daemon refuses. A switch that does not move
 * under the finger reads as broken, but a switch that stays moved after a
 * failed write is worse: it would report a state the daemon does not hold, and
 * the whole point of this control is to show what the agent can actually call.
 *
 * `tool` is the same object the list and `selectedTool` both hold, so mutating
 * `enabled` updates the sidebar row and the detail header together without a
 * refetch.
 */
async function setToolEnabled(tool: Tool, enabled: boolean) {
  if (togglingTools.value.has(tool.name)) return

  const previous = tool.enabled
  tool.enabled = enabled
  togglingTools.value = new Set(togglingTools.value).add(tool.name)

  try {
    await invoke('set_tool_enabled', { name: tool.name, enabled })
    toast.success(`${tool.name} ${enabled ? 'enabled' : 'disabled'}`)
  } catch (e: any) {
    tool.enabled = previous
    toast.error(
      `Could not ${enabled ? 'enable' : 'disable'} ${tool.name}`,
      e instanceof Error ? e.message : String(e),
    )
  } finally {
    const next = new Set(togglingTools.value)
    next.delete(tool.name)
    togglingTools.value = next
  }
}

async function selectTool(tool: Tool) {
  // `confirm` here is the app's composable, not `window.confirm`: it takes
  // options and returns a Promise. Called with a bare string and un-awaited,
  // `!confirm(...)` was `!Promise` — always false — so this guard never
  // prompted and never blocked, and unsaved edits were dropped silently.
  if (hasChanges.value) {
    const discard = await confirm({
      title: 'Discard changes?',
      message: 'This tool has unsaved changes. Switching away will discard them.',
      confirmLabel: 'Discard',
      danger: true,
    })
    if (!discard) return
  }

  selectedTool.value = tool
  creating.value = false
  hasChanges.value = false
  loadingSource.value = true
  activeTab.value = 'details'

  // Load tool details
  try {
    const result = await invoke<{ tool: ToolDetails }>('get_tool', { name: tool.name })
    toolDetails.value = result.tool

    editingTool.value = {
      name: tool.name,
      description: tool.description,
      toolType: 'script',
      code: '',
      outputTarget: 'memory',
    }
    originalCode.value = ''

    // Load source code from tools directory
    try {
      const source = await invoke<{ name: string; source: string; language?: string; path?: string }>('get_tool_source', { name: tool.name })
      if (source && source.source) {
        editingTool.value.code = source.source
        editingTool.value.toolType = source.language === 'yaml' ? 'manifest' : 'script'
        editingTool.value.outputTarget = detectOutputTarget(source.source)
        originalCode.value = source.source
      }
    } catch {
      // Fall back to user tool manager
      try {
        const userTool = await invoke<{ name: string; source: string; language: string } | null>('get_user_tool', { name: tool.name })
        if (userTool && userTool.source) {
          editingTool.value.code = userTool.source
          editingTool.value.toolType = userTool.language === 'yaml' ? 'manifest' : 'script'
          editingTool.value.outputTarget = detectOutputTarget(userTool.source)
          originalCode.value = userTool.source
        }
      } catch {
        // Source not available
      }
    }
  } catch (e) {
    console.error('Failed to load tool details:', e)
    toolDetails.value = null
  } finally {
    loadingSource.value = false
  }
}

function openCreateModal() {
  showCreateModal.value = true
}

function openTemplates() {
  showCreateModal.value = true
}

function createFromTemplate(template: Template) {
  showCreateModal.value = false
  selectedTool.value = null
  toolDetails.value = null
  creating.value = true
  editingTool.value = {
    name: '',
    description: '',
    toolType: template.toolType,
    code: template.code,
    outputTarget: 'memory',
  }
  originalCode.value = ''
  hasChanges.value = true
  activeTab.value = 'code'
}

async function saveTool() {
  saving.value = true
  try {
    if (creating.value) {
      await invoke('create_user_tool', {
        name: editingTool.value.name,
        description: editingTool.value.description,
        source: editingTool.value.code,
      })
      creating.value = false
    } else {
      await invoke('update_user_tool', {
        name: editingTool.value.name,
        description: editingTool.value.description,
        source: editingTool.value.code,
      })
    }
    originalCode.value = editingTool.value.code
    hasChanges.value = false
    await refreshTools()

    // Re-select the tool
    const updated = tools.value.find(t => t.name === editingTool.value.name)
    if (updated) selectTool(updated)
  } catch (e: any) {
    alert('Failed to save: ' + e.toString())
  } finally {
    saving.value = false
  }
}

function confirmDelete() {
  showDeleteModal.value = true
}

async function deleteTool() {
  if (!selectedTool.value) return
  deleting.value = true
  try {
    await invoke('delete_user_tool', { name: selectedTool.value.name })
    selectedTool.value = null
    toolDetails.value = null
    creating.value = false
    await refreshTools()
  } catch (e: any) {
    alert('Failed to delete: ' + e.toString())
  } finally {
    deleting.value = false
    showDeleteModal.value = false
  }
}

function testTool() {
  showTestPanel.value = true
  testResult.value = null
  testInputs.value = {}
  testInputJson.value = '{}'
}

async function runTest() {
  testing.value = true
  testError.value = false
  testResult.value = null
  testDuration.value = null

  const start = Date.now()
  try {
    let input: Record<string, any> = {}
    if (toolDetails.value?.parameters?.properties) {
      input = { ...testInputs.value }
    } else {
      input = JSON.parse(testInputJson.value)
    }

    if (selectedTool.value) {
      // Test existing tool
      testResult.value = await invoke<string>('test_skill', {
        code: editingTool.value.code || '',
        skillType: editingTool.value.toolType,
        input,
      })
    } else if (creating.value) {
      // Test new tool
      testResult.value = await invoke<string>('test_skill', {
        code: editingTool.value.code,
        skillType: editingTool.value.toolType,
        input,
      })
    }
  } catch (e: any) {
    testError.value = true
    testResult.value = e.toString()
  } finally {
    testDuration.value = Date.now() - start
    testing.value = false
  }
}

function onEditorMount(_editor: any) {
  // Editor mounted
}

/** Detect output target from tool source code */
function detectOutputTarget(source: string): 'memory' | 'context' {
  // Match output: "context" or output: 'context' (JS/TS)
  if (/output:\s*["']context["']/.test(source)) return 'context'
  // Match output: context (YAML)
  if (/^output:\s*context\s*$/m.test(source)) return 'context'
  return 'memory'
}

/** Sync the output target dropdown value into the tool source code */
function syncOutputTargetToSource() {
  const code = editingTool.value.code
  const target = editingTool.value.outputTarget
  const isYaml = editingTool.value.toolType === 'manifest'

  if (isYaml) {
    // YAML: add/update/remove `output: context` line
    if (target === 'context') {
      if (/^output:\s*\w+/m.test(code)) {
        editingTool.value.code = code.replace(/^output:\s*\w+/m, 'output: context')
      } else {
        // Add after description line
        editingTool.value.code = code.replace(
          /^(description:.*)/m,
          `$1\noutput: context`
        )
      }
    } else {
      // Remove the output line
      editingTool.value.code = code.replace(/\n?\s*output:\s*(memory|context)\s*$/m, '')
    }
  } else {
    // JS/TS: add/update/remove `output: "context"` field
    if (target === 'context') {
      if (/output:\s*["'](memory|context)["']/.test(code)) {
        editingTool.value.code = code.replace(
          /output:\s*["'](memory|context)["']/,
          'output: "context"'
        )
      } else {
        // Add after description line
        editingTool.value.code = code.replace(
          /(description:\s*["'][^"']*["'],?\s*\n)/,
          `$1  output: "context",\n`
        )
      }
    } else {
      // Remove the output line
      editingTool.value.code = code.replace(/\s*output:\s*["'](memory|context)["'],?\n?/, '\n')
    }
  }
}

// Watch for changes
watch(() => editingTool.value.code, (newCode) => {
  hasChanges.value = newCode !== originalCode.value
})

watch(() => editingTool.value.description, () => {
  if (selectedTool.value && editingTool.value.description !== selectedTool.value.description) {
    hasChanges.value = true
  }
})

watch(() => editingTool.value.outputTarget, () => {
  syncOutputTargetToSource()
  hasChanges.value = editingTool.value.code !== originalCode.value
})

// Load on mount
onMounted(() => {
  refreshTools()
})
</script>
