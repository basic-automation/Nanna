<template>
  <div class="space-y-4">
    <!-- Status Header -->
    <div class="flex items-center justify-between">
      <div class="flex items-center gap-2">
        <div 
          :class="[
            'w-2 h-2 rounded-full',
            isPolling ? 'bg-nanna-success animate-pulse' : 'bg-nanna-text-dim'
          ]"
        />
        <span class="text-sm text-nanna-text-muted">
          {{ isPolling ? 'Live updates active' : 'Updates paused' }}
        </span>
      </div>
      <div class="flex items-center gap-2">
        <UiButton @click="togglePolling" variant="ghost" size="sm">
          <component :is="isPolling ? Pause : Play" class="w-4 h-4" />
        </UiButton>
        <UiButton @click="testAllChannels" :disabled="testing" variant="ghost" size="sm">
          <RefreshCw :class="['w-4 h-4', testing && 'animate-spin']" />
        </UiButton>
      </div>
    </div>

    <!-- Channel Cards -->
    <div class="space-y-3">
      <div
        v-for="channel in channels"
        :key="channel.provider"
        :class="[
          'p-4 rounded-lg border transition-all',
          getStateClasses(channel.connection_state)
        ]"
      >
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-3">
            <span class="text-2xl">{{ getChannelIcon(channel.provider) }}</span>
            <div>
              <h4 class="font-medium text-nanna-text">{{ channel.name }}</h4>
              <p class="text-xs text-nanna-text-dim">{{ channel.details || 'Not configured' }}</p>
            </div>
          </div>
          
          <div class="flex items-center gap-2">
            <!-- Status Badge -->
            <UiBadge :variant="getStatusVariant(channel.connection_state)">
              {{ channel.connection_state }}
            </UiBadge>
            
            <!-- Metrics -->
            <div v-if="channel.configured" class="flex items-center gap-3 text-xs text-nanna-text-muted">
              <!-- Response Time -->
              <div v-if="channel.avg_response_ms" class="flex items-center gap-1" :title="`Avg response: ${channel.avg_response_ms.toFixed(0)}ms`">
                <Clock class="w-3 h-3" />
                <span>{{ channel.avg_response_ms.toFixed(0) }}ms</span>
              </div>
              
              <!-- Queue -->
              <div v-if="channel.queue_depth > 0" class="flex items-center gap-1" title="Messages queued">
                <Inbox class="w-3 h-3" />
                <span>{{ channel.queue_depth }}</span>
              </div>
              
              <!-- Rate Limit -->
              <div v-if="channel.rate_limit_remaining_ms" class="flex items-center gap-1 text-nanna-warning" title="Rate limited">
                <AlertTriangle class="w-3 h-3" />
                <span>{{ formatCooldown(channel.rate_limit_remaining_ms) }}</span>
              </div>
            </div>
          </div>
        </div>
        
        <!-- Expanded Stats (when configured) -->
        <div v-if="channel.configured && showDetails" class="mt-3 pt-3 border-t border-white/[0.04]">
          <div class="grid grid-cols-4 gap-4 text-xs">
            <div>
              <div class="text-nanna-text-muted">Sent/hr</div>
              <div class="text-nanna-text font-medium">{{ channel.messages_sent_hour }}</div>
            </div>
            <div>
              <div class="text-nanna-text-muted">Failed/hr</div>
              <div :class="['font-medium', channel.messages_failed_hour > 0 ? 'text-nanna-error' : 'text-nanna-text']">
                {{ channel.messages_failed_hour }}
              </div>
            </div>
            <div>
              <div class="text-nanna-text-muted">Failures</div>
              <div :class="['font-medium', channel.consecutive_failures > 0 ? 'text-nanna-warning' : 'text-nanna-text']">
                {{ channel.consecutive_failures }}
              </div>
            </div>
            <div>
              <div class="text-nanna-text-muted">Last OK</div>
              <div class="text-nanna-text font-medium">
                {{ channel.last_healthy ? formatTime(channel.last_healthy) : '—' }}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
    
    <!-- Toggle Details -->
    <button 
      @click="showDetails = !showDetails" 
      class="w-full text-center text-xs text-nanna-text-muted hover:text-nanna-text py-2"
    >
      {{ showDetails ? 'Hide details' : 'Show details' }}
    </button>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import { invoke } from '@tauri-apps/api/core'
import { listen, type UnlistenFn } from '@tauri-apps/api/event'
import { Play, Pause, RefreshCw, Clock, Inbox, AlertTriangle } from '@lucide/vue'

interface EnhancedChannelStatus {
  name: string
  provider: string
  configured: boolean
  enabled: boolean
  status: string
  details: string | null
  connection_state: string
  last_healthy: number | null
  consecutive_failures: number
  avg_response_ms: number | null
  messages_sent_hour: number
  messages_failed_hour: number
  queue_depth: number
  queue_retrying: number
  rate_limit_remaining_ms: number | null
}

interface ChannelStatusEvent {
  provider: string
  status: EnhancedChannelStatus
  previous_state: string | null
  timestamp: number
}

const channels = ref<EnhancedChannelStatus[]>([])
const isPolling = ref(false)
const testing = ref(false)
const showDetails = ref(false)
let unlistenStatus: UnlistenFn | null = null

const channelIcons: Record<string, string> = {
  telegram: '✈️',
  discord: '🎮',
  slack: '💼',
  signal: '🔒',
  whatsapp: '💬',
}

onMounted(async () => {
  await loadChannelStatus()
  await startPolling()
})

onUnmounted(async () => {
  if (unlistenStatus) {
    unlistenStatus()
  }
  if (isPolling.value) {
    await invoke('unsubscribe_channel_status')
  }
})

async function loadChannelStatus() {
  try {
    channels.value = await invoke<EnhancedChannelStatus[]>('get_enhanced_channel_status')
  } catch (e) {
    console.error('Failed to load channel status:', e)
  }
}

async function startPolling() {
  try {
    // Listen for status events
    unlistenStatus = await listen<ChannelStatusEvent>('channel-status', (event) => {
      const update = event.payload
      const idx = channels.value.findIndex(c => c.provider === update.provider)
      if (idx >= 0) {
        channels.value[idx] = update.status
      }
    })
    
    // Start backend polling
    await invoke('subscribe_channel_status', { intervalMs: 30000 })
    isPolling.value = true
  } catch (e) {
    console.error('Failed to start status polling:', e)
  }
}

async function togglePolling() {
  if (isPolling.value) {
    await invoke('unsubscribe_channel_status')
    isPolling.value = false
  } else {
    await startPolling()
  }
}

async function testAllChannels() {
  testing.value = true
  try {
    const results = await invoke<Record<string, { success: boolean; message: string }>>('test_all_channels')
    
    // Update channel statuses based on test results
    for (const channel of channels.value) {
      const result = results[channel.provider]
      if (result) {
        channel.connection_state = result.success ? 'connected' : 'degraded'
        channel.details = result.message
        if (result.success) {
          channel.last_healthy = Date.now()
          channel.consecutive_failures = 0
        } else {
          channel.consecutive_failures++
        }
      }
    }
  } catch (e) {
    console.error('Failed to test channels:', e)
  } finally {
    testing.value = false
  }
}

function getChannelIcon(provider: string): string {
  return channelIcons[provider] || '📡'
}

function getStateClasses(state: string): string {
  switch (state) {
    case 'connected':
      return 'bg-nanna-success/5 border-nanna-success/30'
    case 'rate_limited':
      return 'bg-nanna-warning/5 border-nanna-warning/30'
    case 'degraded':
      return 'bg-nanna-warning/5 border-nanna-warning/30'
    case 'unavailable':
    case 'auth_failed':
      return 'bg-nanna-error/5 border-nanna-error/30'
    default:
      return 'glass-panel'
  }
}

function getStatusVariant(state: string): 'success' | 'warning' | 'error' | 'secondary' {
  switch (state) {
    case 'connected':
      return 'success'
    case 'rate_limited':
    case 'degraded':
      return 'warning'
    case 'unavailable':
    case 'auth_failed':
      return 'error'
    default:
      return 'secondary'
  }
}

function formatCooldown(ms: number): string {
  const seconds = Math.ceil(ms / 1000)
  if (seconds < 60) return `${seconds}s`
  return `${Math.floor(seconds / 60)}m ${seconds % 60}s`
}

function formatTime(timestamp: number): string {
  const now = Date.now()
  const diff = now - timestamp
  
  if (diff < 60000) return 'Just now'
  if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`
  if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`
  return new Date(timestamp).toLocaleDateString()
}
</script>
