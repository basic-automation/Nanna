<template>
  <!-- The design's 32px window radius, drawn by the shell on a transparent
       window. Dropped while maximized — a maximized window has no corners. -->
  <div
    class="nui-root flex h-screen flex-col overflow-hidden text-xs leading-normal"
    :class="!isMaximized && 'rounded-[32px]'"
  >
    <div class="flex min-h-0 w-full flex-1 items-start gap-4">
      <!-- ═══ Main menu rail ═══ -->
      <NuiMainMenu
        :items="railItems"
        :bottom-items="bottomRailItems"
        :active-id="activeRailId"
        class="self-stretch"
        @select="onRailSelect"
      />

      <!-- ═══ Chat menu (session list, chat route only) ═══ -->
      <aside v-if="chatPanelOpen" class="flex w-64 shrink-0 flex-col gap-4 self-stretch overflow-clip pb-8 pt-6">
        <div class="flex w-full items-center justify-between pl-4">
          <p class="text-xs leading-normal text-nui-fg">Chats</p>
          <NuiIconButton icon="add" label="New chat" @click="createNewSession" />
        </div>
        <nav class="nui-scroll flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto">
          <SessionItem
            v-for="session in sessions"
            :key="session.id"
            :session="session"
            :is-active="currentSessionId === session.id"
            @select="(s) => { switchSession(s); }"
            @deleted="onSessionDeleted"
            @renamed="onSessionRenamed"
          />
          <div v-if="sessions.length === 0" class="px-4 py-8 text-center text-xs leading-normal text-nui-muted">
            No chats yet
          </div>
        </nav>
      </aside>

      <!-- ═══ Body column ═══ -->
      <div class="flex min-h-0 min-w-0 flex-1 flex-col gap-4 self-stretch">
        <!-- Top bar: workspace select + window controls -->
        <div class="flex w-full shrink-0 items-start gap-4" data-tauri-drag-region>
          <NuiSelect
            :model-value="workspaceSelectValue"
            :options="workspaceOptions"
            label="Workspace"
            icon="workspaces"
            variant="attached"
            class="w-64 shrink-0"
            @update:model-value="onWorkspaceSelect"
          />
          <div class="min-w-0 flex-1 self-stretch" data-tauri-drag-region />
          <NuiWindowControls
            @minimize="minimizeWindow"
            @maximize="toggleMaximizeWindow"
            @close="closeWindow"
          />
        </div>

        <!-- Page content -->
        <main class="min-h-0 w-full flex-1 overflow-hidden">
          <slot />
        </main>
      </div>
    </div>

    <!-- ═══ Bottom status bar ═══ -->
    <NuiStatusBar
      :ui-version="appVersion || undefined"
      :server-version="daemonVersion || undefined"
      :connected="statusBar.tone === 'ok'"
      :status-text="statusBar.text"
      :updating="updating || checking"
      :update-label="updateLabel"
      :update-tooltip="updateTooltip"
      @update="applyUpdate"
    />

    <!-- Notification Center sheet (opened from the rail bell) -->
    <UiSheet :open="notifOpen" side="right" @update:open="notifOpen = $event">
      <NotificationCenter />
    </UiSheet>

    <!-- Workspace Picker Modal -->
    <WorkspacePicker
      v-if="showWorkspacePicker"
      v-model="showWorkspacePicker"
      :open-tab-ids="openTabIds"
      @select="openWorkspaceTab"
    />

    <!-- Close confirmation dialog -->
    <CloseDialog />

    <CommandPalette
      :open="paletteOpen"
      :actions="paletteActions"
      @close="hidePalette"
      @run="onPaletteRun"
    />

    <!-- Global confirmation dialog lives in app.vue — mounting a second copy
         here opens two stacked overlays sharing the useConfirm singleton, and
         the top one swallows clicks aimed at the bottom one (e2e regression). -->

    <!-- First-run onboarding (compressed) -->
    <OnboardingWizard
      :open="showOnboarding"
      :has-api-key="apiKeySet"
      @close="showOnboarding = false"
      @finished="onOnboardingFinished"
    />
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted, onUnmounted, provide, nextTick } from 'vue'
import { invoke } from '@tauri-apps/api/core'
import { listen, type UnlistenFn } from '@tauri-apps/api/event'
import { getCurrentWindow } from '@tauri-apps/api/window'
import { statusBarLabel } from '~/lib/backendLabels'
import { seedChatModel } from '~/composables/useSessionState'
import { useAppUpdater } from '~/composables/useAppUpdater'
import { useStartupGate } from '~/composables/useStartupGate'
import type { PaletteAction } from '~/lib/commandPalette'
import { NAV_ACTIONS, QUICK_ACTIONS } from '~/lib/commandPalette'
import type { NuiRailItem } from '~/components/nui/NuiMainMenu.vue'

// ═══ Main menu rail (nui design: every page gets its own entry) ═══
const { unreadCount, isOpen: notifOpen } = useNotificationCenter()

const railItems = computed<NuiRailItem[]>(() => [
  { id: '/board', icon: 'tasks', label: 'Board' },
  { id: 'chat', icon: 'chat', label: 'Chats' },
  { id: 'notifications', icon: 'notifications', label: 'Notifications', badge: unreadCount.value },
  { id: '/memory', icon: 'memory', label: 'Memory' },
  { id: '/tools', icon: 'toolbox', label: 'Tools' },
  { id: '/channels', icon: 'channels', label: 'Channels' },
  { id: '/logs', icon: 'log', label: 'Logs' },
  { id: '/workspaces', icon: 'workspaces', label: 'Workspaces' },
  { id: '/agents', icon: 'agents', label: 'Agents' },
  { id: '/scheduler', icon: 'scheduler', label: 'Scheduler' },
  { id: '/model-stats', icon: 'model-stats', label: 'Model Stats' },
  { id: '/tool-stats', icon: 'tool-stats', label: 'Tool Stats' },
  { id: '/tool-audit', icon: 'circle-check', label: 'Tool Audit' },
])

const bottomRailItems: NuiRailItem[] = [
  { id: '/settings', icon: 'settings', label: 'Settings' },
  { id: 'tray', icon: 'chevron-down', label: 'Hide to Tray' },
]

const activeRailId = computed(() => {
  if (notifOpen.value) return 'notifications'
  if (route.path === '/' || route.path === '') return 'chat'
  const entry = [...railItems.value, ...bottomRailItems].find(
    item => item.id.startsWith('/') && isNavActive(item.id),
  )
  return entry?.id
})

function onRailSelect(id: string) {
  if (id === 'notifications') {
    notifOpen.value = !notifOpen.value
    return
  }
  if (id === 'chat') {
    toggleChatPanel()
    return
  }
  if (id === 'tray') {
    void hideToTray()
    return
  }
  navigateTo(id)
}

async function hideToTray() {
  try { await invoke('hide_to_tray') } catch (e) { console.error('Failed to hide to tray:', e) }
}

// ═══ Window controls (frameless window — the top bar is the title bar) ═══
async function minimizeWindow() {
  try { await getCurrentWindow().minimize() } catch (e) { console.error('minimize failed:', e) }
}
async function toggleMaximizeWindow() {
  try { await getCurrentWindow().toggleMaximize() } catch (e) { console.error('maximize failed:', e) }
}
async function closeWindow() {
  try { await getCurrentWindow().close() } catch (e) { console.error('close failed:', e) }
}

interface SessionInfo {
  id: string
  name: string
  created_at: string
  updated_at: string
  message_count: number
  workspace_id: string | null
  workspace_name: string | null
  /** Model this chat is pinned to; absent/null = the global `[llm]` default. */
  chat_model?: string | null
}

interface WorkspaceInfo {
  id: string
  name: string
  path: string
  active?: boolean
}

interface Tab {
  type: 'global' | 'workspace'
  workspaceId?: string
}

interface AppConfig {
  theme: string
  model: string
  api_key_set: boolean
  available_models: string[]
}

const route = useRoute()

const { currentVersion: appVersion, updateVersion, checking, updating, updateError, applyUpdate } = useAppUpdater()

const updateLabel = computed(() => {
  if (updating.value) return 'Updating…'
  if (updateVersion.value) return 'Update to v' + updateVersion.value
  return undefined
})
const updateTooltip = computed(() => {
  if (updateError.value) return 'Update failed: ' + updateError.value + ' — click to retry'
  if (updateVersion.value) return 'Download v' + updateVersion.value + ' and restart Nanna'
  return 'Check for updates'
})

const sessions = ref<SessionInfo[]>([])
const currentSessionId = ref<string | null>(null)
const apiKeySet = ref(false)
const showOnboarding = ref(false)
const ONBOARDING_KEY = 'nanna.onboarding.done'
const showWorkspacePicker = ref(false)
const chatPanelOpen = ref(true)

const openWorkspaces = ref<WorkspaceInfo[]>([])
const allWorkspaces = ref<WorkspaceInfo[]>([])
const currentTab = ref<Tab>({ type: 'global' })

let unlistenTrayNewChat: UnlistenFn | null = null
let unlistenCloseRequested: UnlistenFn | null = null
let unlistenResized: UnlistenFn | null = null

/** Tracked so the shell can drop its corner radius while maximized. */
const isMaximized = ref(false)
let unlistenSessionsCleared: UnlistenFn | null = null
let unlistenSessionRenamed: UnlistenFn | null = null
let unlistenWorkspacesChanged: UnlistenFn | null = null

function isNavActive(path: string) {
  return route.path === path || (path !== '/' && route.path.startsWith(path))
}

function toggleChatPanel() {
  if (route.path !== '/') {
    // Not on chat: go there with the panel open.
    chatPanelOpen.value = true
    navigateTo('/')
    return
  }
  chatPanelOpen.value = !chatPanelOpen.value
}

// ═══ Workspace select (top bar) ═══
const GLOBAL_TAB = 'global'
const OPEN_WORKSPACE = '::open'
const MANAGE_WORKSPACES = '::manage'

const workspaceSelectValue = computed(() =>
  currentTab.value?.type === 'workspace' ? (currentTab.value.workspaceId ?? GLOBAL_TAB) : GLOBAL_TAB,
)

const workspaceOptions = computed(() => [
  { value: GLOBAL_TAB, label: 'Global — all chats' },
  ...allWorkspaces.value.map(ws => ({ value: ws.id, label: ws.name || ws.path })),
  { value: OPEN_WORKSPACE, label: 'Open Workspace…' },
  { value: MANAGE_WORKSPACES, label: 'Manage Workspaces' },
])

function onWorkspaceSelect(value: string) {
  if (value === OPEN_WORKSPACE) {
    showWorkspacePicker.value = true
    return
  }
  if (value === MANAGE_WORKSPACES) {
    navigateTo('/workspaces')
    return
  }
  if (value === GLOBAL_TAB) {
    selectTab({ type: 'global' })
    return
  }
  const ws = allWorkspaces.value.find(w => w.id === value)
  if (ws) {
    addWorkspaceTab(ws)
    selectTab({ type: 'workspace', workspaceId: ws.id })
  }
}

const openTabIds = computed(() => openWorkspaces.value.map(w => w.id))

const activeWorkspace = computed(() => {
  if (currentTab.value?.type === 'workspace') {
    return openWorkspaces.value.find(w => w.id === currentTab.value.workspaceId) || null
  }
  return null
})

function addWorkspaceTab(ws: WorkspaceInfo) {
  if (!openWorkspaces.value.some(w => w.id === ws.id)) {
    openWorkspaces.value.push(ws)
    saveTabsToStorage()
  }
}

function selectWorkspaceTab(workspaceId: string) {
  const ws = openWorkspaces.value.find(w => w.id === workspaceId)
  if (!ws) {
    loadOpenWorkspaces().then(() => {
      const found = openWorkspaces.value.find(w => w.id === workspaceId)
      if (found) selectTab({ type: 'workspace', workspaceId })
    })
  } else {
    selectTab({ type: 'workspace', workspaceId })
  }
}

function selectGlobalTab() {
  selectTab({ type: 'global' })
}

provide('currentSessionId', currentSessionId)
provide('sessions', sessions)
provide('activeWorkspace', activeWorkspace)
provide('currentTab', currentTab)
provide('openWorkspaces', openWorkspaces)
provide('addWorkspaceTab', addWorkspaceTab)
provide('selectWorkspaceTab', selectWorkspaceTab)
provide('selectGlobalTab', selectGlobalTab)
provide('selectTab', selectTab)
provide('closeWorkspaceTab', closeWorkspaceTab)
provide('showWorkspacePicker', showWorkspacePicker)

const { checkPermission } = useNotifications()
const { init: initBackend, status: backendStatus, daemonVersion } = useBackend()
const { releasedOffline } = useStartupGate()
const statusBar = computed(() => statusBarLabel(backendStatus.value, apiKeySet.value))

const { bind: bindShortcut } = useShortcuts()
const { open: paletteOpen, toggle: togglePalette, hide: hidePalette } = useCommandPalette()
const { info: toastInfo } = useToast()

const SESSION_SWITCH_LIMIT = 8

const paletteActions = computed((): PaletteAction[] => {
  const sessionActions: PaletteAction[] = sessions.value.slice(0, SESSION_SWITCH_LIMIT).map((s) => ({
    id: `session:switch:${s.id}`,
    label: s.name || 'Untitled chat',
    group: 'Sessions',
    keywords: ['session', 'chat', 'switch', s.id],
    action: `session:switch:${s.id}`,
  }))
  const workspaceActions: PaletteAction[] = openWorkspaces.value.map((w) => ({
    id: `workspace:switch:${w.id}`,
    label: w.name || w.path || w.id,
    group: 'Workspaces',
    keywords: ['workspace', 'project', 'switch', w.id, w.path],
    action: `workspace:switch:${w.id}`,
  }))
  return [...NAV_ACTIONS, ...QUICK_ACTIONS, ...sessionActions, ...workspaceActions]
})

async function onPaletteRun(action: PaletteAction) {
  hidePalette()
  if (action.href) {
    await navigateTo(action.href)
    return
  }
  const act = action.action
  if (!act) return
  if (act === 'new-chat') {
    await createNewSession()
    return
  }
  if (act === 'toggle-chat-panel') {
    toggleChatPanel()
    return
  }
  if (act === 'focus-input') {
    if (route.path !== '/' && route.path !== '') {
      await navigateTo(currentSessionId.value ? `/?session=${currentSessionId.value}` : '/')
    }
    await nextTick()
    window.dispatchEvent(new CustomEvent('nanna:focus-input'))
    window.dispatchEvent(new CustomEvent('nanna:focus-chat-input'))
    return
  }
  if (act === 'stop-generation') {
    window.dispatchEvent(new CustomEvent('nanna:stop-generation'))
    return
  }
  if (act === 'open-settings-models') {
    await navigateTo('/settings?tab=models')
    return
  }
  if (act === 'toggle-live-logs') {
    const key = 'nanna.logs.live'
    let next = false
    try {
      const cur = localStorage.getItem(key)
      const effectiveLive = cur === null ? true : (cur === '1' || cur === 'true')
      next = !effectiveLive
      localStorage.setItem(key, next ? '1' : '0')
    } catch { /* ignore */ }
    window.dispatchEvent(new CustomEvent('nanna:logs-live', { detail: { live: next } }))
    toastInfo(next ? 'Live logs on' : 'Live logs paused')
    return
  }
  if (act === 'toggle-compact-mode') {
    const root = document.documentElement
    const next = !root.classList.contains('density-compact')
    root.classList.toggle('density-compact', next)
    try { localStorage.setItem('nanna.ui.density', next ? 'compact' : 'comfortable') } catch { /* ignore */ }
    toastInfo(next ? 'Compact mode on' : 'Compact mode off')
    return
  }
  if (act.startsWith('session:switch:')) {
    const id = act.slice('session:switch:'.length)
    const session = sessions.value.find((s) => s.id === id)
    if (session) switchSession(session)
    return
  }
  if (act.startsWith('workspace:switch:')) {
    const id = act.slice('workspace:switch:'.length)
    selectWorkspaceTab(id)
  }
}

// Global shortcuts
bindShortcut({
  key: 'k',
  mod: true,
  priority: 50,
  allowInInput: true,
  description: 'Command palette',
  handler: () => { togglePalette() },
})
bindShortcut({
  key: 'n',
  mod: true,
  shift: true,
  priority: 20,
  description: 'New chat',
  handler: () => { void createNewSession() },
})
bindShortcut({
  key: 'l',
  mod: true,
  shift: true,
  priority: 20,
  allowInInput: true,
  description: 'Focus chat input',
  handler: () => {
    if (route.path !== '/' && route.path !== '') {
      void navigateTo(currentSessionId.value ? `/?session=${currentSessionId.value}` : '/')
    }
    // ChatInput listens for this custom event.
    window.dispatchEvent(new CustomEvent('nanna:focus-chat-input'))
  },
})
bindShortcut({
  key: '.',
  mod: true,
  priority: 20,
  allowInInput: true,
  description: 'Stop generation',
  handler: () => {
    window.dispatchEvent(new CustomEvent('nanna:stop-generation'))
  },
})
const { handleClose, loadCloseMode } = useCloseHandler()

const TABS_STORAGE_KEY = 'nanna-workspace-tabs'
const CURRENT_TAB_KEY = 'nanna-current-tab'

onMounted(async () => {
  try {
    if (localStorage.getItem('nanna.ui.density') === 'compact') {
      document.documentElement.classList.add('density-compact')
    }
  } catch { /* ignore */ }
  // Opened with "Open Nanna anyway", init joins the boot in flight, or starts
  // a daemon that is not running, and answers only when that boot ends, which
  // has no deadline. Awaited, it held back everything below (the first load,
  // the listeners, the close handler) for the whole boot. Start it and go on:
  // the attach watcher below loads what needs a daemon once one answers.
  let mode: 'daemon' | 'disconnected' = 'disconnected'
  if (releasedOffline.value) void initBackend()
  else mode = await initBackend()
  console.log(`Nanna running in ${mode} mode`)
  loadTabsFromStorage()
  // Without a daemon the workspace list comes back empty, and loading it would
  // wipe the tabs just restored from storage. The attach watcher below loads
  // it once a daemon answers.
  if (mode === 'daemon') await loadOpenWorkspaces()
  await loadSessions()
  await loadConfig()
  maybeShowOnboarding()

  // Sync restored workspace state with daemon
  if (mode === 'daemon') syncDaemonWorkspace(currentTab.value)
  initialLoadDone = true
  // Attached between init's answer and here: the watcher skipped it.
  if (mode !== 'daemon' && backendStatus.value?.connected) void reloadFromDaemon()

  const urlSessionId = route.query.session as string | undefined
  if (urlSessionId && sessions.value.some(s => s.id === urlSessionId)) {
    currentSessionId.value = urlSessionId
  }

  unlistenTrayNewChat = await listen('tray-new-chat', () => createNewSession())
  unlistenSessionsCleared = await listen('sessions-cleared', async () => {
    await loadSessions()
    currentSessionId.value = sessions.value[0]?.id || null
  })
  unlistenSessionRenamed = await listen<{ id: string, name: string }>('session-renamed', (event) => {
    const { id, name } = event.payload
    const idx = sessions.value.findIndex(s => s.id === id)
    const existing = sessions.value[idx]
    if (existing) sessions.value[idx] = { ...existing, name }
  })
  // The daemon owns workspace registration, so anything registered while this
  // window is open — by a script, a harness, or another client — must show up
  // in the select without a restart.
  unlistenWorkspacesChanged = await listen('workspaces-changed', () => {
    void loadOpenWorkspaces()
  })

  // Named `appWindow`, not `window` — shadowing the global inside an async mount hook is a trap.
  const appWindow = getCurrentWindow()
  unlistenCloseRequested = await appWindow.onCloseRequested(async (event) => {
    event.preventDefault()
    await handleClose()
  })
  try {
    isMaximized.value = await appWindow.isMaximized()
    unlistenResized = await appWindow.onResized(async () => {
      try { isMaximized.value = await appWindow.isMaximized() } catch { /* browser dev */ }
    })
  } catch { /* browser dev — no Tauri window */ }

  await loadCloseMode()
  await checkPermission()
})

onUnmounted(() => {
  unlistenTrayNewChat?.()
  unlistenCloseRequested?.()
  unlistenResized?.()
  unlistenSessionsCleared?.()
  unlistenSessionRenamed?.()
  unlistenWorkspacesChanged?.()
})

// The backend attaches a daemon that answers late (a slow boot, one started
// by hand, a restart) without an app restart. This lets the window catch up
// on what the first load could not get.
let initialLoadDone = false
async function reloadFromDaemon() {
  await loadOpenWorkspaces()
  await loadSessions()
  await loadConfig()
  await syncDaemonWorkspace(currentTab.value)
}
watch(
  () => backendStatus.value?.connected === true,
  (connected, wasConnected) => {
    if (connected && !wasConnected && initialLoadDone) void reloadFromDaemon()
  },
)

watch(() => route.query.session, (newSessionId) => {
  if (typeof newSessionId === 'string' && sessions.value.some(s => s.id === newSessionId)) {
    currentSessionId.value = newSessionId
  }
})

// The chat menu lives on the chat route: restore it when arriving, drop it
// when leaving (the rail's Chat item can still toggle it while on chat).
watch(() => route.path, (path) => {
  chatPanelOpen.value = (path === '/' || path === '')
})

watch(currentTab, async () => {
  await loadSessions()
  currentSessionId.value = sessions.value[0]?.id || null
  if (currentSessionId.value) {
    navigateTo(`/?session=${currentSessionId.value}`)
  } else {
    // No sessions in this workspace/tab — clear stale session from URL
    navigateTo('/')
  }
  saveTabsToStorage()
}, { deep: true })

function loadTabsFromStorage() {
  try {
    const savedTabs = localStorage.getItem(TABS_STORAGE_KEY)
    const savedCurrent = localStorage.getItem(CURRENT_TAB_KEY)
    if (savedTabs) {
      const tabIds: string[] = JSON.parse(savedTabs)
      openWorkspaces.value = tabIds.map(id => ({ id, name: '', path: '' }))
    }
    if (savedCurrent) currentTab.value = JSON.parse(savedCurrent)
  } catch (e) { console.error('Failed to load tabs from storage:', e) }
}

function saveTabsToStorage() {
  try {
    localStorage.setItem(TABS_STORAGE_KEY, JSON.stringify(openWorkspaces.value.map(w => w.id)))
    localStorage.setItem(CURRENT_TAB_KEY, JSON.stringify(currentTab.value))
  } catch (e) { console.error('Failed to save tabs to storage:', e) }
}

async function loadOpenWorkspaces() {
  try {
    const registered = await invoke<WorkspaceInfo[]>('list_workspaces')
    allWorkspaces.value = registered
    const savedIds = openWorkspaces.value.map(w => w.id)

    if (savedIds.length > 0) {
      // Restore from localStorage (match IDs with backend)
      openWorkspaces.value = registered.filter(ws => savedIds.includes(ws.id))
    } else {
      // No localStorage tabs — restore all registered workspaces from DB
      openWorkspaces.value = registered
    }

    // If current tab points to a workspace that no longer exists, fall back to global
    if (currentTab.value?.type === 'workspace') {
      if (!openWorkspaces.value.some(w => w.id === currentTab.value.workspaceId)) {
        currentTab.value = { type: 'global' }
      }
    }

    // Auto-select active workspace if no current tab is set and one is active
    if (currentTab.value?.type === 'global') {
      const activeWs = registered.find(ws => ws.active)
      if (activeWs && openWorkspaces.value.some(w => w.id === activeWs.id)) {
        currentTab.value = { type: 'workspace', workspaceId: activeWs.id }
      }
    }

    saveTabsToStorage()
  } catch (e) { console.error('Failed to load workspaces:', e); openWorkspaces.value = []; allWorkspaces.value = [] }
}

async function loadSessions() {
  try {
    // Global = show ALL sessions (workspaceId = null → list_sessions returns all)
    // Workspace = show only that workspace's sessions (workspaceId = id)
    const workspaceId = currentTab.value?.type === 'workspace' ? (currentTab.value.workspaceId ?? null) : null
    sessions.value = await invoke<SessionInfo[]>('list_sessions', { workspaceId })
    // The rows render their pin from the session-state store, so that the chat
    // header's picker reaches them the moment a pin changes. This reload is the
    // daemon's own answer for every chat at once, and re-seeding it here is
    // what keeps a pin set in another window from being outlived by ours.
    for (const session of sessions.value) seedChatModel(session.id, session.chat_model ?? null)
    if (sessions.value[0] && !currentSessionId.value) currentSessionId.value = sessions.value[0].id
  } catch (e) { console.error('Failed to load sessions:', e) }
}

async function loadConfig() {
  try {
    const config = await invoke<AppConfig>('get_config')
    apiKeySet.value = config.api_key_set
  } catch (e) { console.error('Failed to load config:', e) }
}

function maybeShowOnboarding() {
  try {
    if (typeof localStorage !== 'undefined' && localStorage.getItem(ONBOARDING_KEY) === '1') {
      return
    }
  } catch {
    /* ignore */
  }
  if (!apiKeySet.value) {
    showOnboarding.value = true
  }
}

function onOnboardingFinished() {
  showOnboarding.value = false
  void loadConfig()
}


function selectTab(tab: Tab) {
  currentTab.value = tab
  // Sync active workspace on daemon for tool working directory + context
  syncDaemonWorkspace(tab)
}

async function syncDaemonWorkspace(tab: Tab) {
  try {
    if (tab.type === 'workspace' && tab.workspaceId) {
      await invoke('set_active_workspace', { id: tab.workspaceId })
    } else {
      await invoke('clear_active_workspace')
    }
  } catch (e) {
    console.error('Failed to sync workspace with daemon:', e)
  }
}

function openWorkspaceTab(ws: WorkspaceInfo) {
  if (!openWorkspaces.value.some(w => w.id === ws.id)) openWorkspaces.value.push(ws)
  currentTab.value = { type: 'workspace', workspaceId: ws.id }
  saveTabsToStorage()
  syncDaemonWorkspace(currentTab.value)
}

function closeWorkspaceTab(workspaceId: string) {
  openWorkspaces.value = openWorkspaces.value.filter(w => w.id !== workspaceId)
  if (currentTab.value?.type === 'workspace' && currentTab.value.workspaceId === workspaceId) {
    currentTab.value = { type: 'global' }
  }
  saveTabsToStorage()
}

async function createNewSession() {
  try {
    const workspaceId = currentTab.value?.type === 'workspace' ? currentTab.value.workspaceId ?? null : null
    const session = await invoke<SessionInfo>('create_session', { name: null, workspaceId })
    currentSessionId.value = session.id
    await loadSessions()
    navigateTo(`/?session=${session.id}`)
  } catch (e) { console.error('Failed to create session:', e) }
}

function switchSession(session: SessionInfo) {
  currentSessionId.value = session.id
  navigateTo(`/?session=${session.id}`)
}

function onSessionDeleted(sessionId: string) {
  sessions.value = sessions.value.filter(s => s.id !== sessionId)
  if (currentSessionId.value === sessionId) {
    currentSessionId.value = sessions.value[0]?.id || null
    if (currentSessionId.value) navigateTo(`/?session=${currentSessionId.value}`)
  }
}

function onSessionRenamed(updated: SessionInfo) {
  const idx = sessions.value.findIndex(s => s.id === updated.id)
  if (idx !== -1) sessions.value[idx] = updated
}
</script>
