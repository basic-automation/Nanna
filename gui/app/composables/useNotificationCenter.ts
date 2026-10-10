import { ref, computed } from 'vue'
import { toast } from 'vue-sonner'

export type NotificationType = 'error' | 'warning' | 'info' | 'success'

export interface AppNotification {
  id: string
  type: NotificationType
  title: string
  summary: string
  detail: string
  source: string
  timestamp: number
  sessionId?: string
  read: boolean
  metadata?: Record<string, any>
}

const notifications = ref<AppNotification[]>([])
const isOpen = ref(false)

// De-dupe window: don't fire the same notification within 2 seconds
const recentHashes = new Map<string, number>()
const DEDUPE_WINDOW_MS = 2000

// The summary is part of a notice's identity: two board notices from one
// agent share a title ("New card for you from …") and differ only there, so
// keying on the title alone dropped the second card's notice for good.
function hashNotification(n: Pick<AppNotification, 'type' | 'title' | 'source' | 'summary'>): string {
  return `${n.type}:${n.source}:${n.title}:${n.summary}`
}

export function useNotificationCenter() {
  const unreadCount = computed(() => notifications.value.filter(n => !n.read).length)

  function addNotification(n: Omit<AppNotification, 'id' | 'timestamp' | 'read'>) {
    // De-dupe check
    const hash = hashNotification(n)
    const now = Date.now()
    const lastSeen = recentHashes.get(hash)
    if (lastSeen && now - lastSeen < DEDUPE_WINDOW_MS) {
      return // Skip duplicate
    }
    recentHashes.set(hash, now)

    // Clean old hashes periodically
    if (recentHashes.size > 100) {
      for (const [key, ts] of recentHashes) {
        if (now - ts > DEDUPE_WINDOW_MS * 5) recentHashes.delete(key)
      }
    }

    const notification: AppNotification = {
      ...n,
      id: crypto.randomUUID(),
      timestamp: now,
      read: false,
    }

    // Add to store (newest first)
    notifications.value.unshift(notification)

    // Cap at 200 notifications
    if (notifications.value.length > 200) {
      notifications.value = notifications.value.slice(0, 200)
    }

    // Fire toast
    const toastOpts = {
      description: n.summary,
      duration: n.type === 'error' ? 8000 : 4000,
      action: {
        label: 'View',
        onClick: () => { isOpen.value = true },
      },
    }

    switch (n.type) {
      case 'error':
        toast.error(n.title, toastOpts)
        break
      case 'warning':
        toast.warning(n.title, toastOpts)
        break
      case 'info':
        toast.info(n.title, toastOpts)
        break
      case 'success':
        toast.success(n.title, toastOpts)
        break
    }
  }

  function markRead(id: string) {
    const n = notifications.value.find(n => n.id === id)
    if (n) n.read = true
  }

  function markAllRead() {
    notifications.value.forEach(n => { n.read = true })
  }

  function clearAll() {
    notifications.value = []
  }

  function removeNotification(id: string) {
    notifications.value = notifications.value.filter(n => n.id !== id)
  }

  function copyAsMarkdown(id: string): string {
    const n = notifications.value.find(n => n.id === id)
    if (!n) return ''

    const icon = { error: '❌', warning: '⚠️', info: 'ℹ️', success: '✅' }[n.type]
    const time = new Date(n.timestamp).toLocaleString()

    let md = `## ${icon} ${n.title}\n\n`
    md += `**Time:** ${time}\n`
    md += `**Source:** ${n.source}\n`
    if (n.sessionId) md += `**Session:** ${n.sessionId}\n`
    md += `\n### Summary\n${n.summary}\n`

    if (n.detail && n.detail !== n.summary) {
      md += `\n### Detail\n\`\`\`\n${n.detail}\n\`\`\`\n`
    }

    if (n.metadata && Object.keys(n.metadata).length > 0) {
      md += `\n### Metadata\n`
      for (const [key, value] of Object.entries(n.metadata)) {
        md += `- **${key}:** ${typeof value === 'object' ? JSON.stringify(value) : value}\n`
      }
    }

    return md
  }

  function relativeTime(timestamp: number): string {
    const seconds = Math.floor((Date.now() - timestamp) / 1000)
    if (seconds < 5) return 'just now'
    if (seconds < 60) return `${seconds}s ago`
    const minutes = Math.floor(seconds / 60)
    if (minutes < 60) return `${minutes}m ago`
    const hours = Math.floor(minutes / 60)
    if (hours < 24) return `${hours}h ago`
    const days = Math.floor(hours / 24)
    return `${days}d ago`
  }

  return {
    notifications,
    unreadCount,
    isOpen,
    addNotification,
    markRead,
    markAllRead,
    clearAll,
    removeNotification,
    copyAsMarkdown,
    relativeTime,
  }
}
