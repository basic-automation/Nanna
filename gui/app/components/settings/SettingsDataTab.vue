<template>
  <div class="space-y-6">
    <SettingsSection
      title="Data Management"
      description="Sessions and memories stored on this machine."
    >
      <template #icon>
        <Database class="w-4 h-4 text-nanna-primary" />
      </template>

      <div class="flex items-center justify-between p-3 rounded-lg glass-panel">
        <div>
          <div class="text-sm font-medium text-nanna-text">Chat Sessions</div>
          <div class="text-xs text-nanna-text-dim">{{ sessionCount }} sessions stored</div>
        </div>
        <UiButton @click="confirmClearSessions" variant="destructive" size="sm">
          <Trash2 class="w-4 h-4 mr-1" />
          Clear All
        </UiButton>
      </div>

      <div class="flex items-center justify-between p-3 rounded-lg glass-panel">
        <div>
          <div class="text-sm font-medium text-nanna-text">Memories</div>
          <div class="text-xs text-nanna-text-dim">{{ memoryStats?.total_memories || 0 }} memories stored</div>
        </div>
        <UiButton @click="confirmClearMemories" variant="destructive" size="sm">
          <Trash2 class="w-4 h-4 mr-1" />
          Clear All
        </UiButton>
      </div>
    </SettingsSection>

    <SettingsSection
      title="Data location"
      description="The folder where the daemon keeps its database, memories and logs. Read when the daemon starts."
    >
      <template #icon>
        <FolderOpen class="w-4 h-4 text-nanna-primary" />
      </template>

      <code
        class="block text-xs glass-well text-nanna-accent p-2 rounded font-mono break-all"
        data-testid="data-dir-path"
      >
        {{ dataDir?.effective ?? dataDirError ?? 'Loading…' }}
      </code>
      <p v-if="dataDir" class="text-xs text-nanna-text-dim" data-testid="data-dir-kind">
        {{ dataDirSummary(dataDir) }}
      </p>
      <div class="flex gap-2">
        <UiButton
          @click="chooseDataDir"
          :disabled="!dataDir || savingDataDir"
          variant="secondary"
          size="sm"
          class="flex-1"
          data-testid="data-dir-choose"
        >
          <FolderOpen class="w-4 h-4 mr-1" />
          Choose folder…
        </UiButton>
        <UiButton
          v-if="dataDir?.is_custom"
          @click="resetDataDir"
          :disabled="savingDataDir"
          variant="secondary"
          size="sm"
          class="flex-1"
          data-testid="data-dir-reset"
        >
          <RotateCcw class="w-4 h-4 mr-1" />
          Use the default
        </UiButton>
      </div>
      <p v-if="restartNotice" class="text-xs text-nanna-warning" data-testid="data-dir-notice">
        {{ restartNotice }}
      </p>
    </SettingsSection>

    <SettingsSection
      title="Configuration"
      description="Export or import your config file."
    >
      <template #icon>
        <FileDown class="w-4 h-4 text-nanna-primary" />
      </template>

      <p class="text-sm text-nanna-text-muted">Config file location:</p>
      <code class="block text-xs glass-well text-nanna-accent p-2 rounded font-mono break-all">
        {{ configPath }}
      </code>
      <div class="flex gap-2">
        <UiButton @click="exportConfig" variant="secondary" size="sm" class="flex-1">
          <FileDown class="w-4 h-4 mr-1" />
          Export
        </UiButton>
        <UiButton @click="importConfig" variant="secondary" size="sm" class="flex-1">
          <FileUp class="w-4 h-4 mr-1" />
          Import
        </UiButton>
      </div>
    </SettingsSection>

    <!-- Calm danger zone — always visible, not shouted -->
    <SettingsSection
      danger
      title="Erase data"
      description="These actions permanently remove local data. Export first if you may need it later."
    >
      <template #icon>
        <Trash2 class="w-4 h-4" />
      </template>
      <p class="text-xs text-nanna-text-muted">
        Use the Clear All actions above. Nothing runs until you confirm.
      </p>
    </SettingsSection>

    <SettingsSection title="About Nanna">
      <template #icon>
        <Moon class="w-4 h-4 text-nanna-primary" />
      </template>
      <p class="text-sm text-nanna-text-muted italic mb-1">
        "I am the light that finds you in darkness, the memory that outlives the flesh."
      </p>
      <div class="space-y-2 text-sm">
        <div class="flex justify-between">
          <span class="text-nanna-text-muted">Version</span>
          <span class="text-nanna-text font-mono">0.1.0</span>
        </div>
        <div class="flex justify-between">
          <span class="text-nanna-text-muted">Stack</span>
          <span class="text-nanna-text">Tauri v2 + Nuxt v4 + Rust</span>
        </div>
      </div>
    </SettingsSection>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { invoke } from '@tauri-apps/api/core'
import { open } from '@tauri-apps/plugin-dialog'
import { Database, Trash2, FileDown, FileUp, Moon, FolderOpen, RotateCcw } from '@lucide/vue'
import { useConfirm } from '~/composables/useConfirm'
import { useSettingsPage } from '~/composables/useSettingsPage'
import { dataDirConfirmMessage, dataDirRestartNotice, dataDirSummary, type DataDirInfo } from '~/lib/dataDir'

const store = useSettingsPage()
const { memoryStats, loadMemoryStats, loadSettings, showToast } = store

const { confirm } = useConfirm()

const sessionCount = ref(0)

const configPath = computed(() => {
  if (navigator.platform.includes('Win')) {
    return '%APPDATA%\\clawd\\Nanna\\config\\config.toml'
  } else if (navigator.platform.includes('Mac')) {
    return '~/Library/Application Support/clawd.Nanna/config.toml'
  } else {
    return '~/.config/nanna/config.toml'
  }
})

// Where data lives: the saved setting, and the folder the running daemon
// reports — it reads the setting only at boot, so the two differ until it
// restarts.
const dataDir = ref<DataDirInfo | null>(null)
const dataDirError = ref<string | null>(null)
const savingDataDir = ref(false)
const restartNotice = computed(() => dataDir.value && dataDirRestartNotice(dataDir.value))

onMounted(async () => {
  await Promise.all([loadSessions(), loadDataDir()])
})

async function loadDataDir() {
  try {
    dataDir.value = await invoke<DataDirInfo>('get_data_dir')
  } catch (e: any) {
    dataDirError.value = `Could not read the data location: ${e.message || e}`
  }
}

async function saveDataDir(path: string | null) {
  savingDataDir.value = true
  try {
    dataDir.value = await invoke<DataDirInfo>('set_data_dir', { path })
    showToast('Data location saved — it applies when the daemon restarts', 'success')
  } catch (e: any) {
    showToast(`Could not use that folder: ${e.message || e}`, 'error')
  } finally {
    savingDataDir.value = false
  }
}

async function chooseDataDir() {
  const current = dataDir.value?.effective
  if (!current) return
  const chosen = await open({ directory: true, multiple: false, defaultPath: current })
  if (typeof chosen !== 'string' || chosen === current) return
  const confirmed = await confirm({
    title: 'Change data location',
    message: dataDirConfirmMessage(current, chosen),
    confirmLabel: 'Use this folder',
  })
  if (!confirmed) return
  await saveDataDir(chosen)
}

async function resetDataDir() {
  const current = dataDir.value
  if (!current?.is_custom) return
  const confirmed = await confirm({
    title: 'Use the default data location',
    message: dataDirConfirmMessage(current.effective, current.default),
    confirmLabel: 'Use the default',
  })
  if (!confirmed) return
  await saveDataDir(null)
}

async function loadSessions() {
  try {
    const sessions = await invoke<{ id: string }[]>('list_sessions')
    sessionCount.value = sessions.length
  } catch (e) {
    console.error('Could not load sessions:', e)
  }
}

async function confirmClearSessions() {
  const confirmed = await confirm({
    title: 'Delete All Sessions',
    message: 'Delete all chat sessions? This cannot be undone.',
    confirmLabel: 'Delete All',
    danger: true
  })

  if (!confirmed) return

  try {
    const count = await invoke<number>('clear_all_sessions')
    showToast(`Cleared ${count} sessions`, 'success')
    sessionCount.value = 0
  } catch (e: any) {
    showToast(`Could not clear sessions: ${e.message || e}`, 'error')
  }
}

async function confirmClearMemories() {
  const confirmed = await confirm({
    title: 'Delete All Memories',
    message: 'Delete all memories? This cannot be undone.',
    confirmLabel: 'Delete All',
    danger: true
  })

  if (!confirmed) return

  try {
    // No scope = every scope, which is what "Delete All Memories" promises.
    await invoke('clear_memories')
    showToast('All memories cleared', 'success')
    await loadMemoryStats()
  } catch (e: any) {
    showToast(`Could not clear memories: ${e.message || e}`, 'error')
  }
}

async function exportConfig() {
  try {
    const config = await invoke<string>('export_config')

    const blob = new Blob([config], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'nanna-config.toml'
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)

    showToast('Configuration exported', 'success')
  } catch (e: any) {
    showToast(`Could not export configuration: ${e.message || e}`, 'error')
  }
}

async function importConfig() {
  try {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = '.toml'

    input.onchange = async (e) => {
      const file = (e.target as HTMLInputElement).files?.[0]
      if (!file) return

      // `confirm` is the async dialog: a bare `!confirm(...)` tested a Promise (always

      // truthy), so the import replaced the configuration without waiting for an answer.

      const confirmed = await confirm({

        title: 'Import Configuration',

        message: 'This will replace your current configuration. Continue?',

        confirmLabel: 'Import',

        danger: true,

      })

      if (!confirmed) return

      // Inside the handler's own try: the outer one has returned by the time
      // the file is chosen, so a refused import used to say nothing at all.
      try {
        const content = await file.text()
        await invoke('import_config', { config: content })
        showToast('Configuration imported', 'success')
        await loadSettings()
      } catch (err: any) {
        showToast(`Could not import configuration: ${err?.message || err}`, 'error')
      }
    }

    input.click()
  } catch (e: any) {
    showToast(`Could not import configuration: ${e.message || e}`, 'error')
  }
}
</script>
